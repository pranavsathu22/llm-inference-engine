"""Day 2 — tests for a single attention Head.

Two things worth checking: the shape contract, and the thing that actually matters --
causal masking. A shape check alone can't catch a broken mask (wrong shapes still come
out); the causality test proves position i's output is mathematically independent of
anything at position > i.
"""

import torch

from slipstream.model import Head


def test_head_output_shape():
    head = Head(n_embd=32, head_size=16, block_size=8, dropout=0.0)
    x = torch.randn(2, 8, 32)          # B=2, T=8, n_embd=32
    out = head(x)
    assert out.shape == (2, 8, 16)     # (B, T, head_size)


def test_head_is_causal():
    """Editing a future position must not change an earlier position's output."""
    torch.manual_seed(0)
    head = Head(n_embd=32, head_size=16, block_size=8, dropout=0.0).eval()

    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out_before = head(x)

    x_edited = x.clone()
    x_edited[:, 7, :] = 999.0          # blow up the LAST position only
    with torch.no_grad():
        out_after = head(x_edited)

    # every position except the edited one must be unaffected
    assert torch.allclose(out_before[:, :7, :], out_after[:, :7, :], atol=1e-5)
    # sanity: the edited position's own output SHOULD differ (otherwise the test is vacuous)
    assert not torch.allclose(out_before[:, 7, :], out_after[:, 7, :], atol=1e-5)
