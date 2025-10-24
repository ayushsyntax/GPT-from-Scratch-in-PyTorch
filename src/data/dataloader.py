# dataloader.py

import torch

def get_batch(data, block_size: int, batch_size: int, device: str):
    """
    Generate a small batch of data of inputs x and targets y.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)
