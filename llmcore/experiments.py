"""The three experiments behind the README's findings.

bench.py measures the shipped configuration. This measures the things that
configuration can't answer on its own: how the cache scales past block_size=64,
how loudly a cache bug fails, and what quantization costs in output quality rather
than just in bytes.

Run with: python -m llmcore.experiments
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .bench import time_generation
from .data import load_tokens, train_val_split
from .generate import generate_cached, generate_naive
from .model import GPT, GPTConfig
from .quantize import dequantize_int8, quantize_int8
from .tokenizer import CharTokenizer

CONTEXTS = [64, 128, 256, 512]


def _roundtrip(past_kv):
    """quantize -> dequantize every (k, v) in a cache, preserving its structure."""
    out = []
    for layer in past_kv:
        out.append([
            (dequantize_int8(*quantize_int8(k)), dequantize_int8(*quantize_int8(v)))
            for k, v in layer
        ])
    return out


def scaling(plots_dir: Path) -> None:
    """How does the cache's advantage change with context length?

    The shipped model is capped at block_size=64, which is far too short to show
    what caching is actually for. Timing doesn't depend on the weights, so this
    sweeps untrained models at larger context sizes.

    Run this alone. It is sensitive to competing load on the machine.
    """
    torch.manual_seed(0)
    prompt = torch.tensor([[5]])
    naive_tps, cached_tps = [], []

    print(f"{'context':>8} | {'N':>5} | {'naive tok/s':>11} | {'cached tok/s':>12} | {'speedup':>7}")
    print("-" * 60)
    for block_size in CONTEXTS:
        cfg = GPTConfig(vocab_size=65, block_size=block_size, n_embd=128, n_head=4, n_layer=4)
        model = GPT(cfg).eval()
        n = block_size - 1

        naive = time_generation(generate_naive, model, prompt.clone(),
                                max_new_tokens=n, temperature=0.8, warmup=1, repeats=5)
        cached = time_generation(generate_cached, model, prompt.clone(),
                                 max_new_tokens=n, temperature=0.8, warmup=1, repeats=5)
        naive_tps.append(naive)
        cached_tps.append(cached)
        print(f"{block_size:>8} | {n:>5} | {naive:>11.1f} | {cached:>12.1f} | {cached/naive:>6.2f}x")

    plt.figure()
    plt.plot(CONTEXTS, naive_tps, marker="o", label="naive (no cache)")
    plt.plot(CONTEXTS, cached_tps, marker="o", label="cached")
    plt.xscale("log", base=2)
    plt.xticks(CONTEXTS, [str(c) for c in CONTEXTS])
    plt.xlabel("context length (tokens generated)")
    plt.ylabel("tokens/sec")
    plt.title("Cache advantage grows with context length")
    plt.legend()
    plt.savefig(plots_dir / "scaling.png")
    plt.close()


def silent_failure(model, tok) -> None:
    """Does a position-accounting bug announce itself?

    Decodes one step three ways: correctly, with the positional-embedding offset
    left at 0 (the "forgot the cache exists" mistake), and through an int8 round
    trip. Compares how far each moves the output distribution.
    """
    torch.manual_seed(0)
    prefill = 32
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prefill))

    with torch.no_grad():
        _, _, past_kv = model(prompt, past_kv=None)
        last = prompt[:, -1:]
        ref = model(last, past_kv=past_kv)[0][:, -1, :]

        # same math as GPT.forward, but positions start at 0 instead of the cached length
        x = model.t_embd(last) + model.pos_embd(torch.arange(0, 1))
        for i, block in enumerate(model.stack):
            x, _ = block(x, past_kv=past_kv[i])
        bad = model.lm_head(model.fln(x))[:, -1, :]

        quant = model(last, past_kv=_roundtrip(past_kv))[0][:, -1, :]

    def compare(name, other):
        drift = (ref - other).abs().max().item()
        tv = 0.5 * (torch.softmax(ref, -1) - torch.softmax(other, -1)).abs().sum().item()
        same = ref.argmax(-1).item() == other.argmax(-1).item()
        print(f"  {name:<26} drift {drift:>7.4f}   total-variation {tv:>6.4f}   "
              f"top-1 {'unchanged' if same else 'CHANGED'}")

    print(f"decoding position {prefill}, neither variant raised an exception")
    compare("offset left at 0 (bug)", bad)
    compare("int8 round trip", quant)


def quantization_cost(model, val_data, trials: int = 200, prefill: int = 48) -> None:
    """What does an int8 cache cost in output quality, not just bytes?

    Prefills real validation text, then decodes one step from the clean cache and
    from a cache that has been through int8, and compares.
    """
    torch.manual_seed(0)
    drifts, clean_losses, quant_losses = [], [], []
    top1_agree = 0

    with torch.no_grad():
        for _ in range(trials):
            start = torch.randint(0, len(val_data) - prefill - 2, (1,)).item()
            chunk = val_data[start : start + prefill].unsqueeze(0)
            target = val_data[start + prefill].view(1)

            _, _, past_kv = model(chunk, past_kv=None)
            last = chunk[:, -1:]
            clean = model(last, past_kv=past_kv)[0][:, -1, :]
            quant = model(last, past_kv=_roundtrip(past_kv))[0][:, -1, :]

            drifts.append((clean - quant).abs().max().item())
            top1_agree += int(clean.argmax(-1).item() == quant.argmax(-1).item())
            clean_losses.append(torch.nn.functional.cross_entropy(clean, target).item())
            quant_losses.append(torch.nn.functional.cross_entropy(quant, target).item())

    mean_clean = sum(clean_losses) / len(clean_losses)
    mean_quant = sum(quant_losses) / len(quant_losses)
    print(f"{trials} trials, prefill {prefill}")
    print(f"  max |logit drift|:     mean {sum(drifts)/len(drifts):.5f}, worst {max(drifts):.5f}")
    print(f"  top-1 token unchanged: {top1_agree}/{trials} = {100*top1_agree/trials:.1f}%")
    print(f"  perplexity:            fp32 {math.exp(mean_clean):.3f} -> "
          f"int8 {math.exp(mean_quant):.3f}  "
          f"({100*(math.exp(mean_quant)/math.exp(mean_clean)-1):+.2f}%)")


def main() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    tok = CharTokenizer(open("data/input.txt").read())
    _, val_data = train_val_split(load_tokens("data/input.txt", tok))
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=4)
    model = GPT(cfg)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    print("=== 1. cache advantage vs context length ===")
    scaling(plots_dir)
    print("\n=== 2. how loudly does a cache bug fail? ===")
    silent_failure(model, tok)
    print("\n=== 3. what does an int8 cache cost in quality? ===")
    quantization_cost(model, val_data)


if __name__ == "__main__":
    main()
