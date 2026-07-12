# Candle Rewrite Analysis

## What fork-candle actually uses from Candle

### candle-core (39 494 LOC)
| API | Used in | Purpose |
|-----|---------|---------|
| `Tensor` | universal_model.rs, candle_client.rs, utils.rs, paged_attention/ | Core tensor ops |
| `Device` | universal_model.rs, utils.rs, paged_attention/ | CPU/CUDA/Metal dispatch |
| `DType` | universal_model.rs, utils.rs | Data types (F32, etc.) |
| `quantized::QMatMul` | universal_model.rs | Quantized matmul (from_arc) |
| `quantized::QTensor` | universal_model.rs | Quantized weight storage |
| `quantized::GgmlDType` | universal_model.rs | Quantization type enum |
| `quantized::gguf_file` | universal_model.rs, candle_client.rs | GGUF file reader |
| `IndexOp` | universal_model.rs | Tensor indexing |
| `Module` | universal_model.rs | Trait for forward() |
| `Result` / `Error::Msg` | universal_model.rs, utils.rs | Error handling |
| `Storage` | paged_attention/op.rs | **YOUR PATCH** — needed storage access |
| `utils::cuda_is_available` | candle_client.rs | CUDA check |
| `utils::metal_is_available` | candle_client.rs | Metal check |
| `D` | universal_model.rs | Dimension index helper |

### candle-nn (9 682 LOC)
| API | Used in | Purpose |
|-----|---------|---------|
| `ops::silu` | universal_model.rs:175 | SwiGLU activation |
| `ops::softmax_last_dim` | universal_model.rs:571 | Attention softmax |

**That's it.** Only 2 functions from candle-nn. Everything else (RmsNorm, RoPE, KvCache) is **inlined** in universal_model.rs.

### candle-transformers (83 110 LOC!)
| API | Used in | Purpose |
|-----|---------|---------|
| `generation::LogitsProcessor` | candle_client.rs:1911-1920 | Token sampling |
| `generation::Sampling` | candle_client.rs:1920 | ArgMax / top-k / top-p |
| `utils::apply_repeat_penalty` | candle_client.rs:2160 | Repeat penalty |

**3 things** from 83K lines of candle-transformers.

---

## Summary: What Candle actually provides to fork-candle

| Crate | LOC | Actually used | Coverage |
|-------|----:|--------------|----------|
| candle-core | 39 494 | Tensor, Device, quantized, gguf_file, Storage | ~15% of API |
| candle-nn | 9 682 | silu, softmax_last_dim | ~2% of API |
| candle-transformers | 83 110 | LogitsProcessor, Sampling, apply_repeat_penalty | ~0.5% of API |
| **Total** | **132 286** | **~20 public items** | **<2%** |

## Your patches (Goosidze/candle vs huggingface/candle)

Only 2 files changed, 3 insertions, 3 deletions:

1. **candle-core/src/tensor.rs**: `storage()` and `storage_mut()` changed from `pub(crate)` → `pub`
   - Needed for paged_attention custom CUDA op to access tensor storage directly
   
2. **candle-core/src/variable.rs**: `storage_mut_and_layout()` wrapped in `unsafe {}`
   - Method was made crate-private in upstream; wrapping in unsafe is a workaround

## What needs rewriting / changing

### Must rewrite (fork-candle depends on it deeply)
- **candle-core**: Tensor, Device, Storage, quantized module, gguf_file
- These are the foundation — can't remove, but CAN simplify

### Can inline and remove dependency
- **candle-nn**: Just copy `silu()` and `softmax_last_dim()` into agent-core (already did this pattern for RmsNorm, RoPE)
- **candle-transformers**: Copy `LogitsProcessor`, `Sampling`, `apply_repeat_penalty()` into agent-core
  - Already inlined quantized_nn, rms_norm, rope — same pattern

### Can strip entirely
- candle-examples (36 401 LOC)
- candle-datasets
- candle-book
- candle-onnx
- candle-pyo3
- candle-wasm-*
- All model files in candle-transformers/src/models/ (fork-candle has UniversalModel)
- candle-flash-attn / candle-flash-attn-v3 (custom implementation planned)
- candle-kernels / candle-metal-kernels (rewrite as needed)
