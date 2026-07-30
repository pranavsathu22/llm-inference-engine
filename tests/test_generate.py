import torch
from llmcore.model import GPT, GPTConfig
from llmcore.generate import generate_naive

def test_generate_naive_runs():
    cfg = GPTConfig(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = GPT(cfg)
    idx = torch.tensor([[3]])
    out = generate_naive(model, idx, max_new_tokens=10)
    assert out.shape == (1, 11)   # 1 seed token + 10 generated