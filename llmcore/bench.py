"""Day 11 — benchmarks.

Numbers, not adjectives. Produce the three plots/tables that go in the README:
  1. tokens/sec: generate_naive vs generate_cached, across sequence lengths.
  2. KV-cache bytes: fp16 vs int8 (from quantize.tensor_bytes).
  3. training loss curve (from the history train() returns).

Time with a warmup + several repeats; if you ever run on CUDA, torch.cuda.synchronize()
before/after timing or your numbers are fiction.
"""

from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import load_tokens
from .generate import generate_cached, generate_naive
from .model import GPT, GPTConfig
from .quantize import quantize_int8, tensor_bytes
from .tokenizer import CharTokenizer


def time_generation(fn, *args, warmup: int = 1, repeats: int = 3, **kwargs) -> float:
    """Return tokens/sec for a generation fn. Warm up, then median of `repeats` runs."""
    sync = torch.cuda.is_available()

    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_time = statistics.median(times)
    max_new_tokens = kwargs["max_new_tokens"]
    return max_new_tokens / median_time


def _kv_cache_bytes(past_kv) -> tuple[int, int, int]:
    """(fp32, fp16, int8) total bytes for one full past_kv -- every layer, every head.

    fp32 is what this model actually runs in (no dtype conversion happens anywhere in
    this codebase); fp16 is measured on real .half()-cast tensors, not just fp32/2, so
    it's honest rather than assumed; int8 is quantize_int8's real output.
    """
    fp32_total = fp16_total = int8_total = 0
    for layer_cache in past_kv:
        for k, v in layer_cache:
            fp32_total += tensor_bytes(k, v)
            fp16_total += tensor_bytes(k.half(), v.half())
            qk, sk, zk = quantize_int8(k)
            qv, sv, zv = quantize_int8(v)
            int8_total += tensor_bytes(qk, sk, zk, qv, sv, zv)
    return fp32_total, fp16_total, int8_total


def main() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    tok = CharTokenizer(open("data/input.txt").read())
    data = load_tokens("data/input.txt", tok)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=4)
    model = GPT(cfg)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    # --- 1. tokens/sec: naive vs cached, across generation lengths -----------------
    # Bounded by block_size=64 (positional embeddings only go up to block_size-1;
    # generate_cached crashes past that since, unlike generate_naive, it never
    # truncates old context -- see the Day 8 devlog entry).
    lengths = [8, 16, 32, 48, 63]
    naive_tps, cached_tps = [], []
    prompt = torch.tensor([[tok.encode("T")[0]]])

    print(f"{'N':>4} | {'naive tok/s':>12} | {'cached tok/s':>13} | {'speedup':>8}")
    for n in lengths:
        naive_speed = time_generation(
            generate_naive, model, prompt.clone(), max_new_tokens=n, temperature=0.8
        )
        cached_speed = time_generation(
            generate_cached, model, prompt.clone(), max_new_tokens=n, temperature=0.8
        )
        naive_tps.append(naive_speed)
        cached_tps.append(cached_speed)
        print(f"{n:>4} | {naive_speed:>12.1f} | {cached_speed:>13.1f} | {cached_speed/naive_speed:>7.2f}x")

    plt.figure()
    plt.plot(lengths, naive_tps, marker="o", label="naive (no cache)")
    plt.plot(lengths, cached_tps, marker="o", label="cached")
    plt.xlabel("tokens generated")
    plt.ylabel("tokens/sec")
    plt.title("Generation speed: naive vs KV cache")
    plt.legend()
    plt.savefig(plots_dir / "tokens_per_sec.png")
    plt.close()

    # --- 2. KV-cache bytes: fp32 / fp16 / int8, across sequence length -------------
    seq_lengths = [4, 8, 16, 24, 32, 40, 48, 56, 63]
    fp32_list, fp16_list, int8_list = [], [], []

    print(f"\n{'T':>4} | {'fp32 B':>9} | {'fp16 B':>9} | {'int8 B':>9} | {'fp16/int8':>9}")
    for t in seq_lengths:
        idx = data[:t].unsqueeze(0)
        with torch.no_grad():
            _, _, past_kv = model(idx, past_kv=None)
        fp32_b, fp16_b, int8_b = _kv_cache_bytes(past_kv)
        fp32_list.append(fp32_b)
        fp16_list.append(fp16_b)
        int8_list.append(int8_b)
        print(f"{t:>4} | {fp32_b:>9} | {fp16_b:>9} | {int8_b:>9} | {fp16_b/int8_b:>8.2f}x")

    plt.figure()
    plt.plot(seq_lengths, fp32_list, marker="o", label="fp32")
    plt.plot(seq_lengths, fp16_list, marker="o", label="fp16")
    plt.plot(seq_lengths, int8_list, marker="o", label="int8 (quantized)")
    plt.xlabel("sequence length (tokens)")
    plt.ylabel("KV-cache bytes")
    plt.title("KV-cache memory: fp32 vs fp16 vs int8")
    plt.legend()
    plt.savefig(plots_dir / "kv_cache_bytes.png")
    plt.close()

    # --- 3. training loss curve -----------------------------------------------------
    # Reuses the curve captured during the Day 7 training run rather than retraining
    # from scratch (history isn't persisted separately -- only the plot is).
    existing_curve = Path("loss_curve.png")
    if existing_curve.exists():
        shutil.copy(existing_curve, plots_dir / "loss_curve.png")
        print(f"\nReused existing training loss curve from {existing_curve}")
    else:
        print("\nNo existing loss_curve.png found -- run tests/test_training.py first.")

    print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
