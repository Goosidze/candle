# Candle Gemma 4 Full Support — Fork Work Tracker

**Base**: `eddyb:gemma4-text-fixes` (PR #3608) on top of `huggingface/candle` main
**Branch**: `gemma4-full-support` in `fork-candle-candle/`

## What eddyb's PR #3608 ALREADY implements ✅

### config.rs
- ✅ `attention_k_eq_v: bool` (for 12B/31B where K=V)
- ✅ `vocab_size_per_layer_input: usize`
- ✅ `hidden_size_per_layer_input: usize`
- ✅ `num_kv_shared_layers: usize`
- ✅ `use_double_wide_mlp: bool`

### text.rs (888 lines)
- ✅ **RmsNorm fix**: removed erroneous `+ 1.0` offset
- ✅ **Attention scaling fix**: replaced erroneous `scale` with `1.0` (Gemma uses post-norm)
- ✅ **RotaryEmbedding**: returns `(cos, sin)` instead of applying rope directly
- ✅ **KvSource enum**: `Computed` (normal) vs `Shared` (for shared KV layers)
- ✅ **SharedKvStates**: `for_full` and `for_sliding` — shared KV between computed + shared layers
- ✅ **attention_k_eq_v**: when true, V = K (no v_proj), used by 12B/31B
- ✅ **Double-wide MLP**: layers ≥ `first_kv_shared_layer_idx` get `intermediate_size * 2`
- ✅ **PerLayerInputMixer**: gate → act_fn → multiply → projection → norm → residual add
- ✅ **PerLayerEmbeddings (PLE)**:
  - `embed_tokens_per_layer`: per-token per-layer embeddings [vocab, num_layers × 256]
  - `per_layer_model_projection`: projects hidden states to per-layer space
  - `per_layer_projection_norm`: RMS norm on projection
  - Forward: projection + embedding lookup, scaled, summed, normed
- ✅ **layer_scalar**: loaded per layer, output multiplied by it
- ✅ **V norm**: `v_norm()` — RMS without learned weight, applied to V
- ✅ **Tensor prefix fix**: `vb.clone()` instead of `vb.pp("model")` (mod.rs handles `model.`)

### mod.rs
- ✅ `vb.pp("model")` → `vb.pp("language_model")` handled correctly
- ✅ `forward_embeds` now takes `input_ids` + `xs` (needed for PLE)

### vision.rs + audio.rs
- ✅ `.pp("linear")` suffix added to all linear layers (weight naming fix)

### examples/gemma4/main.rs
- ✅ Appropriate stop tokens (`<eos>` / `<turn>`)

## What's MISSING or needs work 🔴

### 1. No quantized Gemma 4 model
- ❌ No `candle-transformers/src/models/quantized_gemma4.rs`
- llama.cpp has Gemma 4 GGUF support already
- fork-candle's `UniversalModel` in agent-core handles GGUF Gemma 4, but candle-transformers doesn't
- **Priority**: HIGH — needed for GGUF inference path

### 2. No tests / validation against Python reference
- ❌ No parity tests against `transformers` modeling_gemma4_text.py
- eddyb used `sum_all()` for debugging but no formal test
- **Priority**: HIGH — need to verify logits match before submitting PR

### 3. `vocab_size_per_layer_input` / `hidden_size_per_layer_input` — serde defaults
- ⚠️ No `#[serde(default)]` — but ALL current Gemma 4 variants have these fields:
  - E2B: `vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=256`
  - 12B: `vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=0`
  - 31B: `vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=0`
- When `hidden_size_per_layer_input=0`, PLE is effectively disabled (code checks `> 0`)
- **Priority**: LOW — current models all have the fields, but add defaults for robustness

### 4. `store_full_length_kv` logic is complex and untested
- ⚠️ Determines which "computed" layer stores its KV for "shared" layers to reuse
- Logic: `rposition` of same `layer_type` in `layer_types[..first_kv_shared_layer_idx]`
- This is subtle — could have edge cases with different layer_type patterns
- **Priority**: MEDIUM — needs test with actual 12B/31B models

### 5. MoE support (26B-A4B)
- ❌ eddyb mentions "not tried yet"
- Need: expert routing, top-k selection, sparse forward
- Can reference `candle-transformers/src/models/qwen3_moe.rs`
- **Priority**: LOW — large models, less common

### 6. `Tensor::get_on_dim()` — EXISTS in upstream candle ✅
- ✅ `candle-core/src/tensor.rs:2235` — `pub fn get_on_dim<D: Dim>(&self, dim: D, index: usize) -> Result<Tensor>`
- Used in PLE forward: `per_layer_inputs.get_on_dim(2, i)` — gets layer i's per-layer input
- **Priority**: DONE — no action needed

### 7. No `candle-examples/examples/gemma4/` updates for PLE
- ⚠️ Example was updated for stop tokens, but may not test PLE path
- **Priority**: LOW

## Compilation status

```bash
cargo check -p candle-transformers  # ✅ PASSES on eddyb's branch
```

## Gemma 4 model configs (from HuggingFace)

| Field | E2B | 12B | 31B |
|-------|-----|-----|-----|
| `hidden_size` | 1536 | 3840 | 5120 |
| `intermediate_size` | 6144 | 15360 | 20480 |
| `num_hidden_layers` | 35 | 40 | 48 |
| `num_attention_heads` | 8 | 16 | 32 |
| `num_key_value_heads` | 4 | 8 | 8 |
| `num_global_key_value_heads` | 1 | 4 | 4 |
| `attention_k_eq_v` | false | **true** | **true** |
| `vocab_size_per_layer_input` | 262144 | 262144 | 262144 |
| `hidden_size_per_layer_input` | **256** | 0 | 0 |
| `num_kv_shared_layers` | **20** | 0 | 0 |
| `use_double_wide_mlp` | **true** | false | false |

**Key insight**: E2B is the ONLY variant with PLE + shared KV + double-wide MLP.
12B/31B are simpler (no PLE, no shared KV, but have K=V attention).

## Plan for our fork

### Step 1: Verify eddyb's branch compiles
- Checkout `gemma4-full-support`
- `cargo check -p candle-transformers`
- Fix any compilation errors

### Step 2: Add serde defaults for PLE config fields
```rust
fn default_vocab_size_per_layer_input() -> usize { 0 }
fn default_hidden_size_per_layer_input() -> usize { 0 }

#[serde(default = "default_vocab_size_per_layer_input")]
pub vocab_size_per_layer_input: usize,
#[serde(default = "default_hidden_size_per_layer_input")]
pub hidden_size_per_layer_input: usize,
```

### Step 3: Check if `get_on_dim` exists
- If not, implement it or find alternative (narrow + reshape)

### Step 4: Write parity test
- Load `google/gemma-4-E2B-it` (safetensors)
- Run forward pass with known input
- Compare logits against `transformers` reference
- This validates RmsNorm, attention scaling, PLE, shared KV, double-wide MLP

### Step 5: Implement quantized_gemma4.rs
- Port GGUF loading from fork-candle's UniversalModel
- Support Q4_0, Q5_K_M, Q8_0 etc.
- KV cache quant (fix the dequant→requant bug while at it)

### Step 6: Clean up and submit PR to huggingface/candle
- Squash into clean commits
- Add documentation
- Reference issue #3448 and PR #3608

## Files to modify (in candle-transformers)

| File | What to do |
|------|-----------|
| `src/models/gemma4/config.rs` | Add serde defaults for PLE fields |
| `src/models/gemma4/text.rs` | Fix get_on_dim if needed, add comments |
| `src/models/gemma4/mod.rs` | Verify multimodal forward works |
| `src/models/quantized_gemma4.rs` | **NEW** — quantized GGUF support |
| `src/models/gemma4/vision.rs` | Already fixed by eddyb |
| `src/models/gemma4/audio.rs` | Already fixed by eddyb |
