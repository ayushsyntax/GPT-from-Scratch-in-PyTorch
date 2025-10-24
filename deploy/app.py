# deploy/app.py

import torch
import gradio as gr
import json
from pathlib import Path
from src.data.tokenizer import CharTokenizer
from src.model.transformer import GPT

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"
MODEL_PATH = ROOT / "gpt_mini_shakespeare.pth"
METRICS_PATH = ROOT / "evaluation_metrics.json"

# Load text
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
model = GPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Load metrics if available
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    eval_text = (
        f"Negative Log-Likelihood: {metrics['nll']:.4f}\n"
        f"Perplexity: {metrics['perplexity']:.2f}\n"
        f"Bits Per Character: {metrics['bpc']:.4f}\n"
        f"Character Accuracy: {metrics['character_accuracy']*100:.2f}%"
    )
else:
    eval_text = "Evaluation metrics not found."

def generate(prompt: str, length: int = 100) -> str:
    if not prompt.strip():
        prompt = "The king said"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    output = model.generate(context, max_new_tokens=length, temperature=0.8, top_k=50)
    return tokenizer.decode(output[0].tolist())

with gr.Blocks() as demo:
    gr.Markdown("# GPT-Mini: Shakespeare Language Model")
    gr.Markdown(
        "A minimal GPT trained from scratch. Architecture: GPT-2 style with learned positional embeddings and pre-LayerNorm."
    )

    with gr.Tab("Generate"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="The king said")
            length = gr.Slider(10, 200, value=100, label="Length")
        output = gr.Textbox(label="Generated Text", lines=10)
        btn = gr.Button("Generate")
        btn.click(generate, inputs=[prompt, length], outputs=output)

    with gr.Tab("Evaluate"):
        gr.Textbox(label="Validation Metrics", value=eval_text, lines=5, interactive=False)

if __name__ == "__main__":
    demo.launch(share=True)
