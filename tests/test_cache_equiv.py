"""Day 12 — THE test. The KV-cache path must be numerically equivalent to the naive
path: same prompt, same seed => token-identical output. This is the test that proves
your optimization didn't quietly change the model. If it fails, suspect position
offsets or the causal mask in the cached path.

Run with: pytest -q
"""

import torch

from llmcore.model import GPT, GPTConfig
from llmcore.generate import generate_naive, generate_cached


def test_cached_matches_naive_greedy():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=17, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    # greedy (temperature -> 0) so the comparison is deterministic
    out_naive = generate_naive(model, prompt.clone(), max_new_tokens=10, temperature=1e-6)
    out_cached = generate_cached(model, prompt.clone(), max_new_tokens=10, temperature=1e-6)

    assert torch.equal(out_naive, out_cached), "cached generation diverged from naive"
