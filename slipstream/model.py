"""Days 2–5 — the model, built bottom-up.

Head (day 2) -> MultiHeadAttention (day 3) -> Block (day 4) -> GPT (day 5).

Shape convention throughout: B = batch, T = time/sequence, C = n_embd (channels).
Write the shape of every tensor in a comment as you go — shape bugs are 90% of the
pain here, and naming the shapes is how you catch them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # max context length
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0


class Head(nn.Module):
    """Day 2 — one head of self-attention.

    Linear projections to key/query/value, scaled dot-product attention over
    (B, T, head_size), a causal mask so position t only sees <= t, softmax, then a
    weighted sum of values. Register the causal mask as a buffer (it's not a parameter).
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        # TODO(day2): key/query/value Linear(n_embd, head_size, bias=False);
        #             register_buffer("tril", torch.tril(ones(block_size, block_size)))
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embd) -> out: (B, T, head_size)
        # scores = q @ k.transpose(-2,-1) / sqrt(head_size); mask future; softmax; @ v
        # TODO(day2)
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Day 3 — n_head heads in parallel, concatenated then projected back to n_embd."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        # TODO(day3): ModuleList of Head(head_size = n_embd // n_head); output proj Linear(n_embd, n_embd)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concat head outputs on the channel dim, then project. (B,T,C) -> (B,T,C)
        # TODO(day3)
        raise NotImplementedError


class FeedForward(nn.Module):
    """Day 4 — position-wise MLP: Linear -> GELU/ReLU -> Linear, usually 4x inner width."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        # TODO(day4)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(day4)
        raise NotImplementedError


class Block(nn.Module):
    """Day 4 — a transformer block with PRE-norm residuals:
        x = x + attn(ln1(x))
        x = x + ffwd(ln2(x))
    Pre-norm (LN before the sublayer) trains far more stably than post-norm — try both
    and watch the loss if you want to feel why.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        # TODO(day4)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(day4)
        raise NotImplementedError


class GPT(nn.Module):
    """Day 5 — the whole model.

    token embedding + positional embedding -> n_layer Blocks -> final LayerNorm ->
    LM head (Linear n_embd -> vocab_size). Tie the LM head weight to the token
    embedding. forward returns logits, and loss if targets are given.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO(day5)
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) -> logits: (B, T, vocab_size); loss: scalar cross-entropy or None
        # TODO(day5)
        raise NotImplementedError
