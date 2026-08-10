"""Day 6 -- overfit-one-batch sanity check (scratch script, not a pytest test).

If the model can't memorize a single fixed batch, the backward pass or optimizer
wiring is broken -- this is the fastest sanity check before any real training run.
"""

import torch

from llmcore.data import load_tokens, train_val_split
from llmcore.model import GPT, GPTConfig
from llmcore.tokenizer import CharTokenizer
from llmcore.train import train
from llmcore.generate import generate_naive
import matplotlib.pyplot as plt

tok = CharTokenizer(open("data/input.txt").read())
data = load_tokens("data/input.txt", tok)
train_data, val_data = train_val_split(data)

cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=4)
model = GPT(cfg)

history = train(model, train_data, val_data, steps=4000, lr=3e-3, batch_size=8, device="cpu")

step = [h[0] for h in history]
train_loss = [h[1] for h in history]
val_loss = [h[2] for h in history]

plt.figure()
plt.plot(step, train_loss, label="train loss")
plt.plot(step, val_loss, label="val loss")
plt.xlabel("training step")
plt.ylabel("cross-entropy loss")
plt.title("Training loss: TinyShakespeare, char-level GPT")
plt.legend()
plt.savefig("loss_curve.png")

idx = torch.tensor([[tok.encode("T")[0]]])
out = generate_naive(model, idx, max_new_tokens=200)
decoded = tok.decode(out[0].tolist())

print(decoded)