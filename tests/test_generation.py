# test_generation.py

import torch
from src.model.transformer import GPT
from src.data.tokenizer import CharTokenizer

def test_generation():
    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
    model = GPT(vocab_size=tokenizer.vocab_size, embed_dim=64, num_layers=2, num_heads=2, max_seq_len=32)
    context = torch.tensor([[0, 1, 2]])
    out = model.generate(context, max_new_tokens=5)
    assert out.shape == (1, 8)
    print("✅ test_generation passed")
    
if __name__ == "__main__":
    test_generation()

