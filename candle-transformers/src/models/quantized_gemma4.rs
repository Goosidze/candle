//! Gemma 4 quantized model implementation for GGUF inference.
//!
//! Supports E2B, E4B, 12B, and 31B text variants.
//!
//! Key architectural features:
//! - Group-Query Attention (GQA) with per-head Q/K normalization
//! - Hybrid sliding-window + global attention layers
//! - Shared KV cache between computed and shared-KV layers (E2B/E4B)
//! - Per-layer embeddings (PLE / AltUp) for E2B/E4B
//! - Double-wide MLP for shared-KV layers (E2B/E4B)
//! - K=V attention for 12B/31B full-attention layers
//! - Layer-scalar output scaling
//! - Final logit softcapping
//!
//! GGUF metadata keys (prefix: `gemma4`):
//! - `gemma4.block_count`, `gemma4.embedding_length`, `gemma4.feed_forward_length`
//! - `gemma4.attention.head_count`, `gemma4.attention.head_count_kv`
//! - `gemma4.attention.key_length`, `gemma4.attention.value_length`
//! - `gemma4.attention.key_length_swa`, `gemma4.attention.value_length_swa`
//! - `gemma4.attention.sliding_window`, `gemma4.attention.sliding_window_pattern`
//! - `gemma4.attention.shared_kv_layers`
//! - `gemma4.attention.layer_norm_rms_epsilon`
//! - `gemma4.rope.freq_base`, `gemma4.rope.freq_base_swa`
//! - `gemma4.rope.dimension_count`, `gemma4.rope.dimension_count_swa`
//! - `gemma4.final_logit_softcapping`
//! - `gemma4.embedding_length_per_layer_input`
//!
//! References:
//! - [Gemma 4 Blog](https://blog.google/technology/developers/gemma-4/)
//! - [HuggingFace modeling_gemma4_text.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4_text.py)

use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::D;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;

// ── Defaults ────────────────────────────────────────────────────────────────

pub const MAX_SEQ_LEN: usize = 262144;
pub const DEFAULT_ROPE_FREQ_BASE: f32 = 1_000_000.0;
pub const DEFAULT_ROPE_FREQ_BASE_SWA: f32 = 10_000.0;
pub const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
pub const DEFAULT_FINAL_LOGIT_SOFTCAPPING: f64 = 30.0;

// ── QMatMul wrapper ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

// ── Rotary Embedding ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_freq_base: f32, max_seq_len: usize, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq_base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            sin: idx_theta.sin()?,
            cos: idx_theta.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── Proportional Rotary Embedding (for global/full layers) ──────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        head_dim: usize,
        rope_freq_base: f32,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_freq_base as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), device)?;
        let t = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── V norm (RMS without learned weight) ─────────────────────────────────────

fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── Shared KV states ────────────────────────────────────────────────────────

#[derive(Default)]
struct SharedKvStates {
    for_full: Option<(Tensor, Tensor)>,
    for_sliding: Option<(Tensor, Tensor)>,
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvSource {
    Computed {
        attention_wk: QMatMul,
        attention_wv: Option<QMatMul>, // None when K=V (attention_k_eq_v)
        attention_k_norm: RmsNorm,
        num_kv_heads: usize,
        rms_norm_eps: f64,
        kv_cache: Option<(Tensor, Tensor)>,
        store_full_length_kv: bool,
    },
    Shared,
}

#[derive(Debug, Clone)]
struct LayerAttention {
    kv: KvSource,
    attention_wq: QMatMul,
    attention_wo: QMatMul,
    attention_q_norm: RmsNorm,
    num_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    use_flash_attn: bool,
    neg_inf: Tensor,
}

impl LayerAttention {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        sliding_window: Option<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sw) = sliding_window {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sw < i {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        rotary_emb_global: &ProportionalRotaryEmbedding,
        rotary_emb_local: &RotaryEmbedding,
        shared_kv_states: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.attention_wq.forward(xs)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;

        let (k, v) = match &mut self.kv {
            KvSource::Computed {
                ref attention_wk,
                ref mut attention_wv,
                ref attention_k_norm,
                num_kv_heads,
                rms_norm_eps,
                ref mut kv_cache,
                store_full_length_kv,
            } => {
                let n_kv = *num_kv_heads;
                let eps = *rms_norm_eps;
                let k = attention_wk.forward(xs)?;
                let k = k
                    .reshape((b_sz, seq_len, n_kv, self.head_dim))?
                    .transpose(1, 2)?;
                let k = attention_k_norm.forward(&k.contiguous()?)?;

                let v = match attention_wv {
                    Some(wv) => {
                        let v = wv.forward(xs)?;
                        v.reshape((b_sz, seq_len, n_kv, self.head_dim))?
                            .transpose(1, 2)?
                    }
                    None => k.clone(), // K=V when attention_k_eq_v
                };

                // V norm (RMS without learned weight)
                let v = v_norm(&v, eps)?;

                let (q_rot, k_rot) = if self.is_sliding {
                    rotary_emb_local.apply_rotary_emb_qkv(&q, &k, index_pos)?
                } else {
                    rotary_emb_global.apply_rotary_emb_qkv(&q, &k, index_pos)?
                };

                // KV cache
                let (k_cached, v_cached) = match kv_cache {
                    None => (k_rot, v),
                    Some((k_cache, v_cache)) => {
                        if index_pos == 0 {
                            (k_rot, v)
                        } else {
                            let k_cached = Tensor::cat(&[k_cache, &k_rot], 2)?;
                            let v_cached = Tensor::cat(&[v_cache, &v], 2)?;
                            (k_cached, v_cached)
                        }
                    }
                };

                // Store for shared KV layers
                if *store_full_length_kv {
                    let kv = (k_cached.clone(), v_cached.clone());
                    if self.is_sliding {
                        shared_kv_states.for_sliding = Some(kv);
                    } else {
                        shared_kv_states.for_full = Some(kv);
                    }
                }

                *kv_cache = Some((k_cached.clone(), v_cached.clone()));
                (k_cached, v_cached)
            }
            KvSource::Shared => {
                let (k, v) = if self.is_sliding {
                    shared_kv_states.for_sliding.clone().unwrap()
                } else {
                    shared_kv_states.for_full.clone().unwrap()
                };

                // Still need RoPE for Q
                let (q_rot, _) = if self.is_sliding {
                    rotary_emb_local.apply_rotary_emb_qkv(&q, &k, index_pos)?
                } else {
                    rotary_emb_global.apply_rotary_emb_qkv(&q, &k, index_pos)?
                };

                // K is already rotated from the computed layer
                return {
                    let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                    let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

                    let mask = if seq_len == 1 {
                        None
                    } else {
                        Some(self.mask(
                            b_sz, seq_len, index_pos,
                            if self.is_sliding { Some(512) } else { None },
                            xs.dtype(), xs.device(),
                        )?)
                    };

                    let attn_weights = q_rot.matmul(&k.transpose(2, 3)?)?;
                    let attn_weights = match mask {
                        Some(ref mask) => attn_weights.broadcast_add(mask)?,
                        None => attn_weights,
                    };
                    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                    let attn_output = attn_weights.matmul(&v)?;
                    attn_output
                        .transpose(1, 2)?
                        .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
            self.attention_wo.forward(&attn_output)
                };
            }
        };

        let (q_rot, k_rot) = if self.is_sliding {
            rotary_emb_local.apply_rotary_emb_qkv(&q, &k, index_pos)?
        } else {
            rotary_emb_global.apply_rotary_emb_qkv(&q, &k, index_pos)?
        };

        let k = crate::utils::repeat_kv(k_rot, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(
                b_sz, seq_len, index_pos,
                if self.is_sliding { Some(512) } else { None },
                xs.dtype(), xs.device(),
            )?)
        };

        // Attention scale = 1.0 for Gemma (post-norm compensates)
        let attn_weights = q_rot.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = match mask {
            Some(ref mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.attention_wo.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv {
            KvSource::Computed { kv_cache, .. } => {
                *kv_cache = None;
            }
            KvSource::Shared => {}
        }
    }
}

// ── Per-Layer Input Mixer (PLE / AltUp) ─────────────────────────────────────

#[derive(Debug, Clone)]
struct PerLayerInputMixer {
    per_layer_input_gate: QMatMul,
    per_layer_projection: QMatMul,
    post_per_layer_input_norm: RmsNorm,
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: LayerAttention,
    mlp: Mlp,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    layer_scalar: Tensor,
    pli_mixer: Option<PerLayerInputMixer>,
    is_sliding: bool,
    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        rotary_emb_global: &ProportionalRotaryEmbedding,
        rotary_emb_local: &RotaryEmbedding,
        per_layer_input: Option<Tensor>,
        shared_kv_states: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs, index_pos,
            rotary_emb_global, rotary_emb_local,
            shared_kv_states,
        )?;
        let xs = self.post_attention_norm.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let _enter = self.span_mlp.enter();
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_ffn_norm.forward(&xs)?;
        let xs = (residual + xs)?;
        drop(_enter);

        // PLE mixer
        let xs = match (&self.pli_mixer, per_layer_input) {
            (Some(pli), Some(per_layer_input)) => {
                let residual = &xs;
                let gate = pli.per_layer_input_gate.forward(&xs)?;
                let gate = candle_nn::ops::silu(&gate)?;
                let gated = (gate * &per_layer_input)?;
                let proj = pli.per_layer_projection.forward(&gated)?;
                let normed = pli.post_per_layer_input_norm.forward(&proj)?;
                (residual + normed)?
            }
            (None, None) => xs,
            _ => xs, // Graceful: if only one is present, skip PLE
        };

        // Layer scalar
        xs.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Per-Layer Embeddings ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PerLayerEmbeddings {
    hidden_size_per_layer_input: usize,
    num_layers: usize,
    embed_tokens_per_layer: candle_nn::Embedding,
    per_layer_model_projection: QMatMul,
    per_layer_projection_norm: RmsNorm,
}

// ── Model ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: candle_nn::Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    final_logit_softcapping: Option<f64>,
    rotary_emb_global: ProportionalRotaryEmbedding,
    rotary_emb_local: RotaryEmbedding,
    ple: Option<PerLayerEmbeddings>,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "gemma4";

        let md_get = |s: &str| -> Result<gguf_file::Value> {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v.clone()),
            }
        };

        let md_get_or = |s: &str, default: gguf_file::Value| -> Result<gguf_file::Value> {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => Ok(default),
                Some(v) => Ok(v.clone()),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let feed_forward_length = md_get("feed_forward_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_RMS_NORM_EPS as f32) as f64;

        let sliding_window = md_get("attention.sliding_window")
            .and_then(|m| m.to_u32())
            .unwrap_or(512) as usize;

        let shared_kv_layers = md_get("attention.shared_kv_layers")
            .and_then(|m| m.to_u32())
            .unwrap_or(0) as usize;

        let embedding_length_per_layer_input = md_get("embedding_length_per_layer_input")
            .and_then(|m| m.to_u32())
            .unwrap_or(0) as usize;

        let final_logit_softcapping = md_get("final_logit_softcapping")
            .and_then(|m| m.to_f32())
            .map(|v| v as f64)
            .ok();

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQ_BASE);

        let rope_freq_base_swa = md_get("rope.freq_base_swa")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQ_BASE_SWA);

        let key_length_swa = md_get("attention.key_length_swa")
            .and_then(|m| m.to_u32())
            .unwrap_or(256) as usize;

        // Partial rotary factor for global layers: typically 0.25 for Gemma 4
        let partial_rotary_factor = md_get("rope.partial_rotary_factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(0.25) as f64;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;

        // Output norm
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;

        // Output projection (may be tied to embeddings)
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };

        // Rotary embeddings
        let rotary_emb_global = ProportionalRotaryEmbedding::new(
            key_length,
            rope_freq_base,
            partial_rotary_factor,
            MAX_SEQ_LEN,
            device,
        )?;
        let rotary_emb_local = RotaryEmbedding::new(
            key_length_swa,
            rope_freq_base_swa,
            MAX_SEQ_LEN,
            device,
        )?;

        // Determine first KV-shared layer index
        let first_kv_shared_layer_idx = block_count.saturating_sub(shared_kv_layers);

        // Per-layer embeddings (PLE)
        let ple = if embedding_length_per_layer_input > 0 {
            let embed_tokens_per_layer = ct.tensor(reader, "token_embd_per_layer.weight", device)?;
            let embed_tokens_per_layer = embed_tokens_per_layer.dequantize(device)?;
            let vocab_size_per_layer = embed_tokens_per_layer.dim(0)?;
            Some(PerLayerEmbeddings {
                hidden_size_per_layer_input: embedding_length_per_layer_input,
                num_layers: block_count,
                embed_tokens_per_layer: candle_nn::Embedding::new(
                    embed_tokens_per_layer,
                    block_count * embedding_length_per_layer_input,
                ),
                per_layer_model_projection: QMatMul::from_qtensor(
                    ct.tensor(reader, "per_layer_model_projection.weight", device)?,
                )?,
                per_layer_projection_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, "per_layer_projection_norm.weight", device)?,
                    rms_norm_eps,
                )?,
            })
        } else {
            None
        };

        // Build layers
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let blk_prefix = format!("blk.{layer_idx}");

            // Determine if this is a sliding or full attention layer
            // In GGUF, sliding_window_pattern is arr[bool,block_count]
            // For simplicity, use the same pattern as gemma3: (layer_idx + 1) % sliding_window_type > 0
            // But GGUF also stores the pattern explicitly
            let is_sliding = {
                // Try to read the pattern array from metadata
                if let Ok(pattern) = md_get("attention.sliding_window_pattern") {
                    // It's an array of bools
                    match pattern {
                        gguf_file::Value::Array(values) => {
                            values.get(layer_idx)
                                .and_then(|v| v.to_bool().ok())
                                .unwrap_or(true) // default to sliding
                        }
                        _ => true,
                    }
                } else {
                    // Fallback: pattern type 6 (every 6th layer is full)
                    (layer_idx + 1) % 6 > 0
                }
            };

            let layer_head_dim = if is_sliding { key_length_swa } else { key_length };
            let layer_kv_heads = head_count_kv; // Same for all layers in GGUF
            let num_kv_groups = head_count / layer_kv_heads;

            // Determine if this is a KV-shared layer
            let is_kv_shared = layer_idx >= first_kv_shared_layer_idx;

            // Double-wide MLP for shared-KV layers
            let layer_ffn_length = if is_kv_shared {
                feed_forward_length * 2
            } else {
                feed_forward_length
            };

            // Determine if this computed layer should store its KV for shared layers
            // Last computed layer of the same type (sliding/full) stores KV
            let store_full_length_kv = !is_kv_shared && {
                // Check if this is the last layer of its type before shared layers begin
                let same_type_ahead = (layer_idx + 1..first_kv_shared_layer_idx)
                    .any(|i| {
                        let i_sliding = (i + 1) % 6 > 0; // simplified
                        i_sliding == is_sliding
                    });
                !same_type_ahead
            };

            // Attention weights
            let attention_wq = ct.tensor(reader, &format!("{blk_prefix}.attn_q.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{blk_prefix}.attn_output.weight"), device)?;
            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk_prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let kv = if is_kv_shared {
                KvSource::Shared
            } else {
                let attention_wk = ct.tensor(reader, &format!("{blk_prefix}.attn_k.weight"), device)?;
                let attention_k_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{blk_prefix}.attn_k_norm.weight"), device)?,
                    rms_norm_eps,
                )?;

                // Check if V exists separately (attention_k_eq_v)
                let attention_wv = match ct.tensor(reader, &format!("{blk_prefix}.attn_v.weight"), device) {
                    Ok(wv) => Some(QMatMul::from_qtensor(wv)?),
                    Err(_) => None, // K=V
                };

                KvSource::Computed {
                    attention_wk: QMatMul::from_qtensor(attention_wk)?,
                    attention_wv,
                    attention_k_norm,
                    num_kv_heads: layer_kv_heads,
                    rms_norm_eps,
                    kv_cache: None,
                    store_full_length_kv,
                }
            };

            let self_attn = LayerAttention {
                kv,
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                num_heads: head_count,
                num_kv_groups,
                head_dim: layer_head_dim,
                is_sliding,
                use_flash_attn: false, // Quantized path doesn't use flash-attn
                neg_inf: neg_inf.clone(),
            };

            // MLP
            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{blk_prefix}.ffn_gate.weight"), device)?,
                )?,
                feed_forward_up: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{blk_prefix}.ffn_up.weight"), device)?,
                )?,
                feed_forward_down: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{blk_prefix}.ffn_down.weight"), device)?,
                )?,
            };

            // Norms
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk_prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk_prefix}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk_prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk_prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            // Layer scalar
            let layer_scalar = ct.tensor(reader, &format!("{blk_prefix}.layer_scalar.weight"), device)?;
            let layer_scalar = layer_scalar.dequantize(device)?;

            // Per-layer input mixer (only for models with PLE)
            let pli_mixer = if embedding_length_per_layer_input > 0 {
                Some(PerLayerInputMixer {
                    per_layer_input_gate: QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{blk_prefix}.per_layer_input_gate.weight"), device)?,
                    )?,
                    per_layer_projection: QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{blk_prefix}.per_layer_projection.weight"), device)?,
                    )?,
                    post_per_layer_input_norm: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{blk_prefix}.post_per_layer_input_norm.weight"), device)?,
                        rms_norm_eps,
                    )?,
                })
            } else {
                None
            };

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                self_attn,
                mlp,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                layer_scalar,
                pli_mixer,
                is_sliding,
                span_attn,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: candle_nn::Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            final_logit_softcapping,
            rotary_emb_global,
            rotary_emb_local,
            ple,
            span,
            span_output,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let _enter = self.span.enter();

        // Embed tokens with Gemma scaling
        let mut xs = self.tok_embeddings.forward(x)?;
        xs = (xs * (self.embedding_length as f64).sqrt())?;

        // Per-layer inputs for PLE
        let per_layer_inputs: Option<Tensor> = match self.ple.as_ref() {
            None => None,
            Some(ple) => {
                let per_layer_projection = (ple.per_layer_model_projection.forward(&xs)?
                    * (1.0 / (self.embedding_length as f64).sqrt()))?;
                let mut shape = xs.dims().to_vec();
                shape.pop();
                shape.push(ple.num_layers);
                shape.push(ple.hidden_size_per_layer_input);
                let per_layer_projection = per_layer_projection.reshape(shape)?;
                let per_layer_projection = ple.per_layer_projection_norm.forward(&per_layer_projection)?;

                let per_layer_inputs = (ple.embed_tokens_per_layer.forward(x)?
                    * (ple.hidden_size_per_layer_input as f64).sqrt())?;
                let mut reshape_dims: Vec<usize> = x.dims().to_vec();
                reshape_dims.push(ple.num_layers);
                reshape_dims.push(ple.hidden_size_per_layer_input);
                let per_layer_inputs = per_layer_inputs.reshape(reshape_dims)?;

                Some(((per_layer_projection + per_layer_inputs)? * (1.0 / 2.0f64.sqrt()))?)
            }
        };

        let mut shared_kv_states = SharedKvStates::default();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                index_pos,
                &self.rotary_emb_global,
                &self.rotary_emb_local,
                per_layer_inputs
                    .as_ref()
                    .map(|pli| pli.get_on_dim(2, i))
                    .transpose()?,
                &mut shared_kv_states,
            )?;
        }

        let _enter = self.span_output.enter();
        let xs = xs.i((.., seq_len - 1, ..))?;
        let xs = self.norm.forward(&xs)?;
        let logits = self.output.forward(&xs)?;

        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}
