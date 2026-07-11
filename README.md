# llm-inference-engine

A small, from-scratch LLM inference engine. (Importable package: `slipstream`.) A decoder-only transformer implemented
in PyTorch, plus the inference stack that makes generation fast and cheap: a KV
cache, tunable sampling, and an int8-quantized KV cache — each one measured, not
asserted.

The model is the easy half. The point of this repo is the **systems layer on top of
it**: how you go from a correct forward pass to fast, memory-efficient token
generation, and how you *prove* each optimization is both faster and still correct.

## Why it exists

Autoregressive generation recomputes attention over the whole prefix at every step
unless you cache it. Caching trades memory for speed; quantizing the cache trades a
little accuracy for a lot of memory. This repo builds that stack from zero and
benchmarks every rung:

- **Correctness** — the KV-cache path must produce *token-identical* output to the
  full-recompute path (`tests/test_cache_equiv.py`). An optimization that changes the
  answer isn't an optimization.
- **Speed** — tokens/sec, cached vs. uncached (`slipstream/bench.py`).
- **Memory** — measured KV-cache bytes, fp16 vs. int8.

## Layout

```
slipstream/
  tokenizer.py   # text <-> token ids
  data.py        # corpus -> batched (x, y) tensors
  model.py       # Head -> MultiHeadAttention -> Block -> GPT
  train.py       # training loop, checkpointing
  generate.py    # sampling + the KV cache (the payoff)
  quantize.py    # int8 KV cache + memory measurement
  bench.py       # tokens/sec + memory benchmarks, plots
tests/           # shape checks + cache-equivalence
scripts/         # data download
DEVLOG.md        # daily build log — read this to see how it was actually built
```

## Build plan (one concept per commit)

Each box is one day / one commit. The commit message says what broke and why the fix works.

- [ ] **1 — data in.** Tokenizer (char-level to start) + batched data loader. Overfit-able tiny corpus.
- [ ] **2 — one attention head.** Q/K/V, scaled dot-product, causal mask. Understand why the mask and why √d.
- [ ] **3 — multi-head attention.** Split into heads, attend, concat, project.
- [ ] **4 — a transformer block.** FeedForward + residual + pre-LayerNorm.
- [ ] **5 — the full GPT.** Token + positional embeddings, stacked blocks, LM head, weight tying.
- [ ] **6 — training loop.** AdamW + LR warmup. Overfit a single batch to loss ≈ 0 (proves backprop).
- [ ] **7 — real run.** Train on the corpus; sample coherent text. Save loss curve.
- [ ] **8 — KV cache.** Cache K/V during generation. Cached vs. recompute; measure the speedup.
- [ ] **9 — sampling.** Temperature, top-k, top-p.
- [ ] **10 — int8 KV cache.** Quantize the cache; measure the memory drop.
- [ ] **11 — benchmarks.** tokens/sec + KV bytes, fp16 vs int8, with plots in the README.
- [ ] **12 — tests + polish.** Shape tests, cache-equivalence test, per-module writeup.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/download_data.py                 # TinyShakespeare
```

## Results

_Filled in on day 11 — loss curve, tokens/sec (cached vs uncached), KV memory (fp16 vs int8)._
