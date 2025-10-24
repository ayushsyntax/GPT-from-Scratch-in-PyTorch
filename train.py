import os
import torch
import yaml
import random
import numpy as np
from src.data.tokenizer import CharTokenizer
from src.model.transformer import GPT
from src.train.trainer import train_step
from src.train.config import ModelConfig, TrainConfig
from src.utils.debug import show_generation

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Load data
    with open("data/tinyshakespeare.txt", "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Dataset size: {len(data)} characters, vocab size: {tokenizer.vocab_size}")

    # Load config
    config_path = "configs/gpt1_char.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])

    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_cfg.embed_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout
    ).to(train_cfg.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    # Training loop
    for step in range(train_cfg.max_steps):
        loss = train_step(model, data, optimizer, train_cfg)
        if step % train_cfg.eval_interval == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            show_generation(model, tokenizer, device=train_cfg.device)

    # Save model
    torch.save(model.state_dict(), "gpt_mini_shakespeare.pth")
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()