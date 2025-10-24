# export.py

import torch

def export_to_onnx(model, path: str = "gpt_mini.onnx"):
    model.eval()
    model.cpu()
    dummy_input = torch.randint(0, 65, (1, 64), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"}
        },
        opset_version=13
    )
    print(f"✅ Model exported to {path}")
