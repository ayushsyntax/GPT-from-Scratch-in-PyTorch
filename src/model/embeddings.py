# src/model/embeddings.py
import torch.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb.weight[:x.size(1), :].unsqueeze(0)
        return token_emb + pos_emb