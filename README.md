# llmcore

A small, from-scratch LLM inference engine. A decoder-only transformer implemented in
PyTorch, plus the inference stack that makes generation fast and cheap: a KV cache,
tunable sampling, and an int8-quantized KV cache. Each one measured, not asserted.

The model is the easy half. The point of this repo is the **systems layer on top of
it**: how you go from a correct forward pass to fast, memory-efficient token
generation, and how you *prove* each optimization is both faster and still correct.

## Why it exists

Autoregressive generation recomputes attention over the whole prefix at every step
unless you cache it. Caching trades memory for speed. Quantizing the cache trades a
little accuracy for a lot of memory. This repo builds that stack from zero and
benchmarks every rung:

- **Correctness.** The KV-cache path must produce *token-identical* output to the
  full-recompute path (`tests/test_cache_equiv.py`). An optimization that changes the
  answer isn't an optimization.
- **Speed.** Tokens/sec, cached vs. uncached (`llmcore/bench.py`).
- **Memory.** Measured KV-cache bytes, fp16 vs. int8.

## Layout

```
llmcore/
  tokenizer.py   # text <-> token ids
  data.py        # corpus -> batched (x, y) tensors
  model.py       # Head -> MultiHeadAttention -> Block -> GPT
  train.py       # training loop, checkpointing
  generate.py    # sampling + the KV cache (the payoff)
  quantize.py    # int8 KV cache + memory measurement
  bench.py       # tokens/sec + memory benchmarks, plots
tests/           # shape, causality, cache-equivalence, sampling, quantization tests
scripts/         # data download
plots/           # committed benchmark output (see Results)
DEVLOG.md        # daily build log, dated, read this to see how it was actually built
```

### Per-module notes

- **`tokenizer.py`.** Char-level `CharTokenizer`. The vocab is just the sorted set of
  unique characters in the corpus, so it's language-agnostic (see `data.py`'s
  docstring on swapping corpora).
- **`data.py`.** `get_batch` samples random windows and shifts by one for next-token
  targets. `load_tokens`/`train_val_split` are corpus-agnostic. Nothing here is
  Shakespeare-specific.
- **`model.py`.** `Head`'s causal mask uses an absolute-position slice
  (`self.mask[T_cached:T_cached+T_new, :T_total]`), so one formula handles no-cache,
  prefill, and cached decode without branching. That's the piece that makes
  `Head`/`MultiHeadAttention`/`Block`/`GPT.forward` all cache-transparent. `GPT`
  derives the KV-cache position offset from `past_kv`'s own shape instead of a
  separately-tracked argument, so it can't drift out of sync.
- **`train.py`.** Standard AdamW loop. `estimate_loss` always restores
  `model.train()` before returning, even though it evaluates in `.eval()` mode.
- **`generate.py`.** `generate_naive` is the deliberately slow correctness reference;
  it recomputes everything every step. `generate_cached` prefills once, then decodes
  one token at a time, feeding only the newest token each step (see DEVLOG.md for why
  the whole growing sequence must never be re-fed once caching is in play).
  `sample_next` composes temperature, top-k, and top-p filtering before the final
  multinomial draw.
- **`quantize.py`.** Per-token affine int8 quantization (`quantize_int8`/
  `dequantize_int8`). The grouping axis and its accuracy/memory tradeoff vs.
  per-channel grouping are documented in the module docstring. Not yet wired into
  live generation; see Known limitations below.
- **`bench.py`.** Every number here is measured, not projected: real wall-clock
  timing (warmup + median of repeats) and real tensor byte counts, not theoretical
  estimates.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/download_data.py                 # TinyShakespeare
```

## Results

All numbers below are measured on the actual trained checkpoint (`n_embd=128,
n_head=4, n_layer=4, block_size=64`, TinyShakespeare, 4000 training steps), not
projected. Reproduce with `python -m llmcore.bench` from the repo root.

### Training loss: TinyShakespeare, char-level GPT

![Training loss: TinyShakespeare, char-level GPT](plots/loss_curve.png)

### Generation speed: naive vs. KV cache

![Generation speed: naive vs KV cache](plots/tokens_per_sec.png)

| tokens generated | naive tok/s | cached tok/s | speedup |
|---:|---:|---:|---:|
| 8  | 586.3 | 769.4 | 1.31x |
| 16 | 658.1 | 744.1 | 1.13x |
| 32 | 553.9 | 666.5 | 1.20x |
| 48 | 369.1 | 716.5 | 1.94x |
| 63 | 454.2 | 738.9 | 1.63x |

Swept up to `block_size=64`; `generate_cached` can't exceed that (see Known
limitations below). An earlier informal, single-run check with no warmup and no
repeats showed cached losing. That was measurement noise, not a real result. See
DEVLOG.md for why a proper benchmark methodology (warmup + median-of-repeats)
matters, and what changed between the two measurements.

### KV-cache memory: fp32 vs. fp16 vs. int8

![KV-cache memory: fp32 vs fp16 vs int8](plots/kv_cache_bytes.png)

At this model's `head_size=32`: int8 is about 3.2x smaller than fp32, about 1.6x
smaller than fp16. The ratio is constant at every sequence length, since the
per-token quantization overhead (one `scale` and `zero_point` per token) doesn't
grow with sequence length. Not the naive "int8 = 4x" claim; see DEVLOG.md for the
`(group_size * 4) / (group_size + 8)` formula this follows and why smaller groups
dilute the savings.

## Known limitations

Written down honestly rather than glossed over. Both are real properties of this
implementation, not bugs, and both are the kind of thing a production system has to
solve that this toy-scale project didn't need to.

- **`generate_cached` can't exceed `block_size` tokens.** Positional embeddings are a
  fixed-size lookup table (`nn.Embedding(block_size, n_embd)`). `generate_naive`
  avoids this by silently cropping old context every step, but `generate_cached` is
  built to never forget anything, so it has nowhere to go once the position offset
  exceeds `block_size`. Real systems solve this with relative or rotary position
  encodings (RoPE) or a sliding-window cache eviction policy. Out of scope here.
- **Quantization is measured but not deployed.** `quantize.py` is fully correct and
  tested in isolation (round-trip error, memory savings), and `bench.py` uses it to
  measure what a quantized cache would cost. But nothing in `generate_cached`'s
  actual decode path calls `quantize_int8`/`dequantize_int8`. Wiring it in (store
  quantized, dequantize just before the attention matmul) and measuring the resulting
  perplexity/accuracy impact is the natural next step. Not done yet.
