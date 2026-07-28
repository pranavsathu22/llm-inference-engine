"""Day 11 — benchmarks.

Numbers, not adjectives. Produce the three plots/tables that go in the README:
  1. tokens/sec: generate_naive vs generate_cached, across sequence lengths.
  2. KV-cache bytes: fp16 vs int8 (from quantize.tensor_bytes).
  3. training loss curve (from the history train() returns).

Time with a warmup + several repeats; if you ever run on CUDA, torch.cuda.synchronize()
before/after timing or your numbers are fiction.
"""

from __future__ import annotations


def time_generation(fn, *args, warmup: int = 1, repeats: int = 3, **kwargs) -> float:
    """Return tokens/sec for a generation fn. Warm up, then median of `repeats` runs."""
    # TODO(day11)
    raise NotImplementedError


def main() -> None:
    # TODO(day11): load a checkpoint, run the comparisons, save plots to ./plots/, print a table.
    raise NotImplementedError


if __name__ == "__main__":
    main()
