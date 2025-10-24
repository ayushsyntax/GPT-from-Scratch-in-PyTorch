# trainer.py

import torch
from src.data.dataloader import get_batch

def train_step(model, data, optimizer, config, device):
    model.train()
    xb, yb = get_batch(data, model.max_seq_len, config.batch_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()



