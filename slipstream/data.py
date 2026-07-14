"""Day 1 — data loading.

Turn one long token stream into random (x, y) training batches, where y is x shifted
right by one (next-token prediction). No fancy dataset class needed — a function that
samples random windows is enough and easier to reason about.
"""

from __future__ import annotations

import torch
from tokenizer import CharTokenizer


def load_tokens(path: str, tokenizer) -> torch.Tensor:
    """Read the corpus file, encode it, return a 1-D LongTensor of token ids."""
    # TODO(day1)
    txt = open(path).read()
    return torch.tensor(tokenizer.encode(txt), dtype=torch.long)


def train_val_split(data: torch.Tensor, frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the token stream into (train, val). Contiguous split, not shuffled."""
    # TODO(day1)
    n = int(len(data) * frac)
    return (data[:n], data[n:])


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch.

    Returns:
        x: (batch_size, block_size)   token ids
        y: (batch_size, block_size)   x shifted by one (the targets)
    Pick batch_size random start positions; each row is a window of length block_size.
    """
    # TODO(day1)
    indices = torch.randint(0, len(data) - block_size, (batch_size,))
    x = []
    y = []

    for idx in indices:
        x.append(data[idx: idx + block_size])
        y.append(data[idx + 1: idx + block_size + 1])
    
    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)

    return (x, y)
