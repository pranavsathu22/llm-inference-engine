"""Day 12 — shape contracts. Cheap tests that catch the bugs that waste the most time.

Run with: pytest -q   (or: python -m pytest tests/test_shapes.py)
Fill these in as the corresponding modules come online; xfail/skip until then.
"""

import torch

from slipstream.model import GPT, GPTConfig


def test_gpt_output_shape():
    """GPT(idx) logits must be (B, T, vocab_size)."""
    cfg = GPTConfig(vocab_size=17, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is None


def test_loss_is_scalar_with_targets():
    """With targets, forward returns a scalar loss."""
    cfg = GPTConfig(vocab_size=17, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, targets=idx)
    assert loss.ndim == 0
