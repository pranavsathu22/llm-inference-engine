"""Days 6–7 — training.

Day 6: get the loop working and OVERFIT A SINGLE BATCH to loss ~ 0. If a model can't
memorize one batch, your backward pass or loss is wrong — this is the fastest sanity
check in all of deep learning, do it before any real run.

Day 7: train on the full corpus, track train/val loss, checkpoint, sample text.
"""

from __future__ import annotations

import torch
from . import data as d


@torch.no_grad()
def estimate_loss(model, data, block_size: int, batch_size: int, iters: int, device: str) -> float:
    """Average loss over `iters` random batches (model in eval mode)."""
    # TODO(day6)
    model.eval()
    avg = float(0)

    for i in range(iters):
        x, y = d.get_batch(data, block_size, batch_size, device)
        logits, loss = model(x, targets=y)
        avg += loss.item()

    model.train()

    return avg / iters


def train(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    block_size: int,
    device: str,
    eval_every: int = 200,
    ckpt_path: str = "model.pt",
) -> list[tuple[int, float, float]]:
    """AdamW loop. Returns a history of (step, train_loss, val_loss) for the loss curve.

    Consider a short LR warmup. Save a checkpoint at the end (and maybe on best val).
    """
    # TODO(day6-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []

    for step in range(steps):

        x, y = d.get_batch(train_data, block_size, batch_size, device)

        logits, loss = model(x, targets=y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % eval_every == 0:
            train_loss = estimate_loss(model, train_data, block_size, batch_size, 10, device)
            val_loss = estimate_loss(model, val_data, block_size, batch_size, 10, device)
            history.append((step, train_loss, val_loss))
            print(history[-1])

    torch.save(model.state_dict(), ckpt_path)

    return history