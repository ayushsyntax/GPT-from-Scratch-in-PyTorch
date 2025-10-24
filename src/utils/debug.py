# debug.py


import torch

def show_generation(model, tokenizer, prompt: str = "The king said", device="cpu"):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8)
    print("Generated Text:\n", tokenizer.decode(generated[0].tolist()))