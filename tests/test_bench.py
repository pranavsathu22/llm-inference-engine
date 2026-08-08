"""Day 11 -- test for time_generation's mechanics (not the full main() pipeline,
which needs a real checkpoint + corpus -- covered by actually running bench.py)."""

import time

from llmcore.bench import time_generation


def test_time_generation_returns_positive_tokens_per_sec():
    def fake_generate(n, *, max_new_tokens):
        time.sleep(0.01)
        return n

    tps = time_generation(fake_generate, 1, max_new_tokens=20, warmup=1, repeats=2)
    assert tps > 0
