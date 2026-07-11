"""Day 10 — int8 KV cache.

The KV cache is usually the memory bottleneck at long context. Quantizing it to int8
roughly quarters its footprint vs fp32 (halves vs fp16). Use per-tensor (or, better,
per-token/per-channel) affine quantization:

    scale = (max - min) / 255
    q     = round((x - min) / scale)        # uint8/int8
    x_hat = q * scale + min                 # dequant

Store q (int8) + scale + zero-point instead of the fp values. Measure the ACTUAL bytes
of each representation — don't project it — and report accuracy/perplexity impact.
This is the same low-precision-inference idea as the HQQ KV cache in your research repo,
stripped to its essentials so it's legible on its own.
"""

from __future__ import annotations

import torch


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x (fp) -> (q int8, scale, zero_point). Choose your grouping axis and document it."""
    # TODO(day10)
    raise NotImplementedError


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8 -> fp tensor approximating the original."""
    # TODO(day10)
    raise NotImplementedError


def tensor_bytes(*tensors: torch.Tensor) -> int:
    """Total real storage of these tensors: sum(t.numel() * t.element_size()). Use this
    to compare fp16 cache bytes vs (int8 q + scales + zero-points) bytes."""
    # TODO(day10)
    raise NotImplementedError
