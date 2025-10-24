import torch
import math
import json
from src.data.tokenizer import CharTokenizer
from src.model.transformer import GPT

# Load data and model
with open("data/tinyshakespeare.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

model = GPT(tokenizer.vocab_size)
model.load_state_dict(torch.load("gpt_mini_shakespeare.pth"))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

val_data = data[int(0.9 * len(data)):]  # use last 10% as validation

@torch.no_grad()
def evaluate(model, data, block_size=256, batch_size=32, max_batches=200):
    """Compute NLL, perplexity, BPC, and character accuracy on validation data."""
    total_loss, total_acc, total_tokens = 0, 0, 0

    for _ in range(max_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)

        logits, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_acc += (torch.argmax(logits, dim=-1) == y).sum().item()
        total_tokens += x.numel()

    avg_nll = total_loss / total_tokens
    return {
        "nll": avg_nll,
        "perplexity": math.exp(avg_nll),
        "bpc": avg_nll / math.log(2),
        "character_accuracy": total_acc / total_tokens,
        "total_tokens_evaluated": total_tokens
    }

# Evaluate and report
metrics = evaluate(model, val_data)

print("\n" + "="*50)
print("GPT-MINI EVALUATION REPORT (Shakespeare)")
print("="*50)
print(f"NLL:           {metrics['nll']:.4f}")
print(f"Perplexity:    {metrics['perplexity']:.2f}")
print(f"BPC:           {metrics['bpc']:.4f}")
print(f"Char Accuracy: {metrics['character_accuracy']*100:.2f}%")
print(f"Tokens Eval'd: {metrics['total_tokens_evaluated']:,}")
print("="*50)

with open("evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to evaluation_metrics.json")
