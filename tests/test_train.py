"""Day 6 -- tests for the training loop.

The real proof this all works is the overfit-one-batch script (tests/test_training.py,
run manually -- not a pytest test since it's meant to be watched, not asserted on with
a hard threshold). These tests check the mechanical contracts: train() runs without
error, returns clean floats (not tensors), and loss actually decreases on a tiny corpus.
"""

import torch

from llmcore.model import GPT, GPTConfig
from llmcore.train import estimate_loss, train


def _tiny_setup():
    torch.manual_seed(0)
    data = torch.randint(0, 20, (2000,))
    train_data, val_data = data[:1800], data[1800:]
    cfg = GPTConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    return model, train_data, val_data


def test_estimate_loss_returns_float():
    model, train_data, _ = _tiny_setup()
    loss = estimate_loss(model, train_data, block_size=16, batch_size=4, iters=5, device="cpu")
    assert isinstance(loss, float)


def test_estimate_loss_restores_train_mode():
    """estimate_loss must leave the model in train mode, not stuck in eval."""
    model, train_data, _ = _tiny_setup()
    assert model.training
    estimate_loss(model, train_data, block_size=16, batch_size=4, iters=5, device="cpu")
    assert model.training


def test_train_runs_and_loss_decreases():
    model, train_data, val_data = _tiny_setup()
    history = train(
        model, train_data, val_data,
        steps=100, lr=3e-3, batch_size=4, device="cpu",
        eval_every=25, ckpt_path="tests/_scratch_ckpt.pt",
    )
    assert all(isinstance(h[1], float) and isinstance(h[2], float) for h in history)
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss     # loss should have gone down, not stayed flat/diverged
