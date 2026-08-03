"""Days 8–9 — generation: the KV cache (the payoff) and sampling.

Day 8 — the KV cache. Naive generation re-runs the model over the ENTIRE prefix every
step: O(T^2) work to emit T tokens. With a cache, each new token only computes its own
Q and appends its K,V to a running (per-layer, per-head) buffer — O(T) total. The
correctness bar: cached generation must emit TOKEN-IDENTICAL output to the naive path
for the same seed (that's what tests/test_cache_equiv.py checks). If they diverge, your
position handling or your mask is wrong.

Day 9 — sampling. temperature (scale logits), top-k (keep k largest), top-p / nucleus
(keep the smallest set whose prob mass >= p). Greedy is temperature -> 0.

NOTE: to support the cache cleanly you'll extend model.forward to (a) accept a past KV
per layer and (b) accept a position offset for the positional embedding. Decide that
interface here, then thread it back into model.py.
"""

from __future__ import annotations

import torch


def sample_next(logits: torch.Tensor, *, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """logits: (B, vocab) for the last position -> (B, 1) sampled token ids.

    Apply temperature, then top_k and/or top_p filtering, softmax, multinomial sample.
    """
    # TODO(day9)
    logits = logits / temperature
    prob = torch.softmax(logits, dim=1)
    sample = torch.multinomial(prob, num_samples=1)

    return sample

@torch.no_grad()
def generate_naive(model, idx: torch.Tensor, max_new_tokens: int, **sample_kwargs) -> torch.Tensor:
    """Day 8 baseline — NO cache. Every step re-runs the model on the whole (growing)
    sequence, cropped to block_size. Slow on purpose; it's the correctness reference."""
    # TODO(day8)
    for i in range(max_new_tokens):
        idx_cropped = idx[:, -model.cfg.block_size:]
        logits, _, _ = model(idx_cropped)
        next_token = sample_next(logits[:, -1, :], **sample_kwargs)
        idx = torch.cat([idx, next_token], dim=1)

    return idx

@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, **sample_kwargs) -> torch.Tensor:
    """Day 8 — with a KV cache. Prefill the prompt once to build the cache, then each
    step feeds only the newest token + the past KV. Must match generate_naive token for
    token under the same seed."""
    # prefill: run the whole prompt once, build the initial cache
    logits, _, past_kv = model(idx, past_kv=None)
    next_token = sample_next(logits[:, -1, :], temperature=1.0, top_k=None, top_p=None)
    idx = torch.cat([idx, next_token], dim=1)

    # decode: feed ONLY the newest token each step, reusing + growing the cache
    for i in range(max_new_tokens - 1):
        logits, _, past_kv = model(next_token, past_kv=past_kv)
        next_token = sample_next(logits[:, -1, :], **sample_kwargs)
        idx = torch.cat([idx, next_token], dim=1)

    return idx
