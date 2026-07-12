# Candle Rewrite Plan

## Strategy: Progressive extraction, not big-bang rewrite

The goal is to **extract only what fork-candle needs** from the Candle crate tree,
simplify it, and eventually drop the Goosidze/candle fork entirely.

## Phase 0: Map exact dependency surface (DONE)
See ANALYSIS.md — only ~20 public items from 132K LOC are actually used.

## Phase 1: Eliminate candle-nn and candle-transformers dependencies

These are trivial — just inline the 5 used functions:

### 1a. Inline candle-nn (2 functions)
- [ ] Copy `ops::silu()` into `agent-core/src/llm/universal_model.rs` (or a new `ops.rs`)
- [ ] Copy `ops::softmax_last_dim()` into same
- [ ] Remove `candle-nn = "0.11"` from agent-core/Cargo.toml
- [ ] Remove `candle-nn/cuda` and `candle-nn/metal` from feature flags
- [ ] Verify: cargo check with and without `--features cuda`

### 1b. Inline candle-transformers (3 items)
- [ ] Copy `generation::LogitsProcessor` + `Sampling` into `agent-core/src/llm/`
  - Located at: candle-transformers/src/generation.rs
- [ ] Copy `utils::apply_repeat_penalty()` into `agent-core/src/llm/utils.rs`
  - Located at: candle-transformers/src/utils.rs
- [ ] Remove `candle-transformers = "0.11"` from agent-core/Cargo.toml
- [ ] Remove `candle-transformers/cuda` and `candle-transformers/metal` from feature flags
- [ ] Verify: cargo check with and without `--features cuda`

**Result**: 2 of 3 Candle crates gone. Only candle-core remains.

## Phase 2: Patch candle-core upstream instead of forking

The fork exists only for 2 changes (storage pub, storage_mut_and_layout unsafe).
These can be contributed upstream or worked around:

### Option A: PR to huggingface/candle
- [ ] Open PR: make `storage()` / `storage_mut()` public (needed for custom CUDA ops)
- [ ] If accepted: drop fork, point Cargo.toml to upstream
- [ ] If rejected: keep fork, but only candle-core needs it

### Option B: Work around in agent-core
- [ ] Use `unsafe` transmute / raw pointer access to Storage
- [ ] Or restructure paged_attention to not need direct Storage access
- [ ] This is ugly but eliminates the fork entirely

### Option C: Minimal fork (recommended)
- [ ] Strip fork to ONLY candle-core (delete all other crates)
- [ ] Apply the 2 patches
- [ ] Point Cargo.toml patch to the minimal fork
- [ ] Much easier to maintain than full fork

## Phase 3: Simplify candle-core (if we own it)

If we maintain our own candle-core:

### 3a. Strip unused code
- [ ] Remove examples, tests that don't apply
- [ ] Remove backends we don't use (Metal? keep if macOS support needed)
- [ ] Remove quantization formats we don't need
- [ ] Simplify Device enum for our use case

### 3b. Fix KV cache quantization
- [ ] Add native quantized KV cache ops (no dequant→requant roundtrip)
- [ ] This is the #1 performance bug — Q4/Q5 KV stripped to dense
- [ ] Look at how llama.cpp handles KV quant (ggml_type_traits)

### 3c. Add Flash Attention
- [ ] candle-flash-attn already exists in upstream
- [ ] Integrate into candle-core or as separate crate
- [ ] Wire through UniversalModel attention path

### 3d. Add MoE support
- [ ] Currently hard error in UniversalModel
- [ ] Need: expert routing, top-k selection, sparse forward
- [ ] Can reference candle-transformers/src/models/mixtral.rs, qwen3_moe.rs

## Phase 4: (Optional) Merge candle-core into agent-core

If candle-core is stripped to only what we need:
- Move quantized/ and gguf_file/ into agent-core/src/llm/
- Move Tensor/Device glue into agent-core
- One crate, one compilation unit
- Extreme but eliminates all cross-crate versioning issues

---

## Immediate next action

**Phase 1a + 1b**: Inline the 5 functions and remove 2 crate dependencies.
This is zero-risk, no behavior change, and drops 92K LOC of dependency.

Want me to start? Which phase first?
