import math
import torch
from llmcore.model import GPT, GPTConfig

cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=2)
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 8))

logits, loss, _ = model(idx)
print(logits.shape, loss)  
targets = torch.randint(0, cfg.vocab_size, (2, 8))          # (2, 8, 65), None

logits, loss, _ = model(idx, targets=targets)
print(loss.item(), math.log(cfg.vocab_size))   # should be roughly close, both ~4.17