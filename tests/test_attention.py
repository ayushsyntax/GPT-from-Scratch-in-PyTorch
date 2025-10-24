# test_attention.py

import torch
from src.model.attention import CausalSelfAttention

def test_causal_attention_shape():
    embed_dim = 128
    num_heads = 4
    max_seq_len = 10
    batch_size = 2
    seq_len = 5

    attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = attn(x)
    assert out.shape == x.shape
    print("test_causal_attention_shape passed")

def test_no_future_leak():
    embed_dim = 64
    num_heads = 2
    max_seq_len = 8
    attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
    x = torch.randn(1, 4, embed_dim)
    out1 = attn(x)

    # Perturb future token
    x2 = x.clone()
    x2[0, 3, :] += 10.0
    out2 = attn(x2)

    # Past tokens (0,1,2) should be unchanged
    assert torch.allclose(out1[0, :3, :], out2[0, :3, :], atol=1e-6)
    print("test_no_future_leak passed")

if __name__ == "__main__":
    test_causal_attention_shape()
    test_no_future_leak()

