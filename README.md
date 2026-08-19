# llmcore

A char-level GPT and the inference stack built on top of it: a KV cache, temperature/top-k/top-p
sampling, and int8 KV-cache quantization.

The transformer itself is the scaffolding. What I was actually after was the layer above it, where
the interesting questions live. Does this optimization change the model's output? By how much does
it actually help? Both of those turn out to be easy to get wrong, and I got both wrong at least
once here before measuring properly.

Three things came out of it that I didn't expect going in.

## 1. The cache is provably equivalent, not approximately equivalent

`generate_cached` and `generate_naive` produce token-identical output under greedy decoding. Not
"close enough." Identical. `tests/test_cache_equiv.py` asserts it, and I verified the same property
layer by layer during the build (split a sequence into prefill plus one-token-at-a-time decode,
thread the cache through, compare against the all-at-once run: max difference `~1e-7`, which is
float non-associativity and nothing else).

The piece that made this work cleanly was the mask. Instead of branching on whether a cache exists,
`Head` slices its causal mask by absolute position:

```python
T_cached = k.shape[1] - q.shape[1]
causal_mask = self.mask[T_cached : T_cached + T_new, :T_total]
```

One formula covers all three cases. No cache reduces to the original `[:T, :T]` slice. Single-token
decode produces a row of all ones, which is correct because a new token is allowed to see its whole
history. Prefill gets ordinary causal masking among the new tokens. `GPT.forward` derives the
position offset from `past_kv`'s own shape rather than tracking it separately, so the two can't
disagree.

## 2. My first benchmark said caching made things slower. It was wrong.

An early single-run timing check showed cached generation at `0.52x`, i.e. losing badly to the naive
path. I nearly wrote that up as a real finding about Python overhead dominating at small scale.

It was noise. Once `bench.py` did a warmup pass and took a median of repeats, the result inverted:

![Generation speed: naive vs KV cache](plots/tokens_per_sec.png)

| tokens generated | naive tok/s | cached tok/s | speedup |
|---:|---:|---:|---:|
| 8  | 586.3 | 769.4 | 1.31x |
| 16 | 658.1 | 744.1 | 1.13x |
| 32 | 553.9 | 666.5 | 1.20x |
| 48 | 369.1 | 716.5 | 1.94x |
| 63 | 454.2 | 738.9 | 1.63x |

Cached wins everywhere, `1.13x` to `1.94x`. The naive line is visibly the noisier one, which is the
tell: it does more work per step, so it has more variance to hide in. Sweep capped at `block_size=64`
for a reason covered under Limitations.

## 3. int8 does not give you 4x, and the gap is predictable

The marketing number for int8 quantization is 4x smaller than fp32. The real number for this model
is `3.2x`, and for fp16 it's `1.6x`:

![KV-cache memory: fp32 vs fp16 vs int8](plots/kv_cache_bytes.png)

The missing factor is the `scale` and `zero_point` stored alongside every group. Those stay float32,
so with per-token grouping they cost a fixed 8 bytes per group no matter how small the group is.
Sweeping group size from 8 to 2048 showed the ratio tracks `(group_size * 4) / (group_size + 8)`
exactly, approaching 4x asymptotically and never arriving. At this model's `head_size=32` that lands
on 3.2x. The ratio also stays flat across sequence length, since group size doesn't change as the
cache grows.

## Training

For completeness, the model this was all measured on. Char-level, `n_embd=128`, `n_head=4`,
`n_layer=4`, `block_size=64`, 4000 steps on TinyShakespeare.

![Training loss: TinyShakespeare, char-level GPT](plots/loss_curve.png)

It generates recognizable words and picks up the script formatting (all-caps speaker names on their
own line) without producing coherent English, which is about right for this size.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/download_data.py                 # TinyShakespeare
python -m pytest tests/ --ignore=tests/test_training.py   # 22 tests
python -m llmcore.bench                         # regenerates every plot above
```

## Layout

```
llmcore/
  model.py       # Head -> MultiHeadAttention -> Block -> GPT, all cache-transparent
  generate.py    # generate_naive (reference) vs generate_cached, plus sample_next
  quantize.py    # per-token affine int8, with the grouping tradeoff in the docstring
  bench.py       # every number here is measured, not projected
  train.py       # AdamW loop
  tokenizer.py   # char-level, vocab is just the corpus's unique characters
  data.py        # random windows, targets shifted by one
tests/           # shapes, causality, cache equivalence, sampling, quantization
DEVLOG.md        # dated build log with the bugs left in
```

`DEVLOG.md` is the honest version of how this went, including the things that broke. The
backward-before-forward crash, the softmax normalizing over the wrong axis, the `sample_kwargs` that
were silently discarded and made the cache look broken when it wasn't.

## Limitations

Two real ones, both consequences of choices I made rather than bugs.

`generate_cached` can't exceed `block_size` tokens. Positional embeddings are a fixed lookup table,
and the cached path never forgets anything, so it runs off the end of that table. `generate_naive`
sidesteps this by silently cropping old context every step. The fixes are rotary embeddings or a
sliding-window eviction policy, neither of which I built.

Quantization is measured but not deployed. `quantize.py` is correct and tested in isolation, and
`bench.py` uses it to measure what a quantized cache would cost, but nothing in the live decode path
calls it. Wiring it in means storing quantized K/V and dequantizing right before the attention
matmul, then measuring the perplexity hit. That's the obvious next thing.
