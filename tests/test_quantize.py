"""Day 10 -- tests for int8 quantization of the KV cache.

Grouping axis: per-token (reduce over the last dim, head_size, keeping B and T) --
finer-grained than per-tensor, more resistant to outliers, documented here per the
quantize_int8 docstring's request.
"""

import torch

from llmcore.quantize import dequantize_int8, quantize_int8, tensor_bytes


def test_quantize_dtype_is_int8():
    x = torch.randn(2, 5, 8) * 3
    q, scale, zero_point = quantize_int8(x)
    assert q.dtype == torch.int8


def test_reconstruction_error_bounded_and_nonzero():
    """Quantization is lossy -- error should be small but NOT exactly zero,
    and bounded by roughly scale/2 (the max possible rounding error per value)."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8) * 3
    q, scale, zero_point = quantize_int8(x)
    x_hat = dequantize_int8(q, scale, zero_point)

    assert not torch.equal(x, x_hat)
    max_err = (x - x_hat).abs().max().item()
    assert max_err <= scale.max().item() / 2 + 1e-6


def test_constant_group_no_nan_or_inf():
    """A group where every value is identical would divide by zero without the
    scale clamp -- confirm it doesn't, and that reconstruction is still exact."""
    x = torch.full((1, 1, 8), 100.0)
    q, scale, zero_point = quantize_int8(x)
    x_hat = dequantize_int8(q, scale, zero_point)

    assert not torch.isnan(x_hat).any()
    assert not torch.isinf(x_hat).any()
    assert torch.allclose(x, x_hat, atol=1e-3)


def test_tensor_bytes_matches_manual_calculation():
    x = torch.randn(2, 5, 8)
    q, scale, zero_point = quantize_int8(x)

    expected = q.numel() * q.element_size() + \
        scale.numel() * scale.element_size() + \
        zero_point.numel() * zero_point.element_size()
    assert tensor_bytes(q, scale, zero_point) == expected
