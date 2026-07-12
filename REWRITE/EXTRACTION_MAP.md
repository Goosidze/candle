# Candle Rewrite — Source Extraction Map

## Files to extract from Candle for inlining into agent-core

### From candle-nn (then drop the crate)
| Function | Candle Source | Lines | Has CUDA? | Has Metal? |
|----------|--------------|------|-----------|------------|
| `ops::silu()` | candle-nn/src/ops.rs:40 | 3 | No (delegates to Tensor::silu) | No |
| `ops::softmax_last_dim()` | candle-nn/src/ops.rs:285-437 | ~153 | **Yes** | **Yes** |

**Note**: `softmax_last_dim` is a CustomOp1 with CUDA+Metal backends. Inlining it requires
either:
- (a) Keep the CustomOp with all backends — complex, need cuda_backend/metal_backend imports
- (b) Replace with `Tensor::softmax(D::Minus1)` — simpler, Candle core already has this
- (c) Use `candle_nn::ops::softmax_last_dim` as-is but from a local copy

**Recommendation**: Option (b). `Tensor::softmax()` in candle-core already works on all backends.
`softmax_last_dim` is just an optimized version with a custom CUDA kernel. Test if Tensor::softmax
is fast enough — if yes, drop the custom op entirely.

### From candle-transformers (then drop the crate)
| Item | Candle Source | Lines | Dependencies |
|------|--------------|-------|-------------|
| `Sampling` enum | candle-transformers/src/generation/mod.rs:10-20 | 11 | None |
| `LogitsProcessor` struct + impl | candle-transformers/src/generation/mod.rs:22-158 | 137 | `candle_nn::ops::softmax_last_dim`, `candle_nn::sampling::gumbel_softmax` |
| `apply_repeat_penalty()` | candle-transformers/src/utils.rs:30-47 | 18 | None |
| `build_causal_mask()` | candle-transformers/src/utils.rs:14-20 | 7 | None (already inlined in universal_model) |
| `repeat_kv()` | candle-transformers/src/utils.rs:49-58 | 10 | None (already inlined in universal_model) |

**Note**: `LogitsProcessor::sample_f()` calls `candle_nn::ops::softmax_last_dim` and 
`candle_nn::sampling::gumbel_softmax`. If we drop candle-nn:
- `softmax_last_dim` → replace with `Tensor::softmax(D::Minus1)`
- `gumbel_softmax` → either inline it too, or drop GumbelSoftmax variant (fork-candle doesn't use it)

### From candle-core (must keep, but can simplify)
| Module | LOC | fork-candle uses |
|--------|----:|-----------------|
| tensor.rs | 3116 | Core — must keep |
| quantized/mod.rs | 1043 | QMatMul, QTensor, GgmlDType — must keep |
| quantized/gguf_file.rs | 638 | GGUF reader — must keep |
| quantized/k_quants.rs | 2841 | Quantization types — keep if loading GGUF |
| quantized/cuda.rs | 1130 | CUDA quantized ops — keep with cuda feature |
| cuda_backend/mod.rs | 2728 | CUDA backend — keep with cuda feature |
| cuda_backend/device.rs | 837 | CUDA device — keep |
| storage.rs | 849 | Storage access (YOUR PATCH) — keep |
| device.rs | 518 | Device enum — keep |
| dtype.rs | ~200 | DType — keep |
| layout.rs | ~400 | Layout — keep |
| op.rs | 1204 | Tensor ops — keep |
| backprop.rs | 809 | Autograd — keep (used by Module trait) |
| shape.rs | 634 | Shape — keep |
| lib.rs | ~300 | Re-exports — keep |
| **Subtotal needed** | **~16 600** | |
| **Not needed** | **~22 900** | cpu_backend, metal_backend, pickle, safetensors, npy, conv, sort, streaming... |

### Can be stripped from candle-core if we own it
| Module | LOC | Why not needed |
|--------|----:|---------------|
| cpu_backend/ | 3327+ | Low-level CPU kernels (keep only if CPU inference needed) |
| metal_backend/ | 2378+ | Metal GPU (macOS only) |
| pickle.rs | 841 | Python pickle format |
| safetensors.rs | 652 | safetensors format (we use GGUF) |
| npy.rs | ~200 | NumPy format |
| conv.rs | ~200 | 2D conv (vision models only) |
| sort.rs | ~150 | Sorting |
| streaming.rs | ~100 | Streaming |
| accelerate.rs | ~50 | macOS Accelerate |
| mkl.rs | ~50 | Intel MKL |

---

## Action Order

1. **Inline `silu`** — trivial, 3 lines (already just `xs.silu()`)
2. **Replace `softmax_last_dim`** with `Tensor::softmax(D::Minus1)` — test perf
3. **Inline `LogitsProcessor` + `Sampling`** — 137 lines, replace candle_nn deps
4. **Inline `apply_repeat_penalty`** — 18 lines, zero deps
5. **Drop candle-nn and candle-transformers** from Cargo.toml + patch section
6. **Simplify candle-core fork** — strip unused modules
