# config.py

from dataclasses import dataclass, field
import torch

@dataclass
class ModelConfig:
    vocab_size: int = 65
    embed_dim: int = 128
    num_layers: int = 6
    num_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3  
    max_steps: int = 2000
    eval_interval: int = 200
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
