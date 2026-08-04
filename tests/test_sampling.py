"""Day 9 -- tests for sample_next's temperature/top_k/top_p filtering."""

import torch

from llmcore.generate import sample_next


def test_top_k_1_is_greedy():
    """top_k=1 should always pick the single highest-logit index, every draw."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5, 3.0]])
    picks = {sample_next(logits.clone(), top_k=1).item() for _ in range(20)}
    assert picks == {1}


def test_top_p_excludes_low_probability_tail():
    """A small top_p should never sample the clearly lowest-probability token."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5, 3.0]])   # index 3 is the lowest logit
    picks = {sample_next(logits.clone(), top_p=0.5).item() for _ in range(50)}
    assert 3 not in picks


def test_no_filtering_keeps_full_support():
    """With top_k=top_p=None, every token should remain reachable."""
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5, 3.0]])
    picks = {sample_next(logits.clone()).item() for _ in range(200)}
    assert picks == {0, 1, 2, 3, 4}
