"""Day 6 -- overfit-one-batch sanity check (scratch script, not a pytest test).

If the model can't memorize a single fixed batch, the backward pass or optimizer
wiring is broken -- this is the fastest sanity check before any real training run.
"""

import torch

from llmcore.data import get_batch, load_tokens, train_val_split
from llmcore.model import GPT, GPTConfig
from llmcore.tokenizer import CharTokenizer

tok = CharTokenizer(open("data/input.txt").read())
data = load_tokens("data/input.txt", tok)
train_data, val_data = train_val_split(data)

cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_embd=64, n_head=4, n_layer=2)
model = GPT(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

x, y = get_batch(train_data, block_size=32, batch_size=8)   # ONE fixed batch, sampled once

for step in range(300):
    logits, loss = model(x, targets=y)   # same x, y every single iteration
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(step, loss.item())