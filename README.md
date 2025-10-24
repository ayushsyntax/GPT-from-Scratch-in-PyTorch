---

```markdown
# GPT-Mini
*A compact GPT-style language model built from scratch, trained on Shakespeare.*

---

### Demo

<video src="demo.mp4" controls width="600"></video>  
*Generates Shakespearean text in real time.*

---

## Overview

**GPT-Mini** is a decoder-only Transformer implemented in **PyTorch**, trained on the complete works of Shakespeare (~1.1M characters).  

It learns **character-level language modeling**, capturing voice, structure, and rhythm from Shakespeare’s plays and poetry.  

---

## Model Workflow

```

Prompt → Tokenize → Embed → [Decoder ×6] → Linear → Softmax → Next Character

```

### Components

1. **Tokenizer**
   - Character-level: each character → unique token  
   - No subword or BPE tokenization  

2. **Embeddings**
   - Token embedding + **learned positional embeddings** (GPT-style)  

3. **Decoder Block** (repeated 6×)
   - **Pre-LayerNorm** → Causal Self-Attention → Residual  
   - **Pre-LayerNorm** → Feedforward (4× width, GELU) → Residual  

4. **Output**
   - Linear projection tied to token embeddings  
   - Softmax for next-character probabilities  

5. **Generation**
   - Autoregressive, supports **temperature** and **top-k sampling**  

---

## Architecture Diagram

```

+--------+     +---------+     +-----------+
| Prompt | --> | Token   | --> | Embedding |
+--------+     | Encoder |     +-----------+
+---------+           |
v
+-----------+
| Decoder  |
| Block ×6 |
+-----------+
|
v
+-----------+
| Linear    |
| Softmax   |
+-----------+
|
v
+-----------+
| Next Char |
+-----------+

```

---

## Model Specifications

| Component         | Details |
|------------------|---------|
| Architecture      | Decoder-only Transformer |
| Layers            | 6 |
| Embedding Dim     | 128 |
| Attention Heads   | 4 |
| Context Length    | 256 |
| Vocabulary Size   | 65 (character-level) |
| Parameters        | 1.23M |
| Positional Encoding | Learned embeddings |
| Training Steps    | 10,000 (~18 min on GPU) |

---

## Results

| Metric             | Value |
|-------------------|-------|
| Perplexity         | 3.02 |
| Character Accuracy | 64.9% |
| NLL                | 1.105 |
| BPC                | 1.594 |

> Evaluated on held-out Shakespeare text. Metrics stored in `evaluation_metrics.json`.

---

## Project Structure

```

gpt-mini/
├── src/model/          # attention.py, transformer.py, embeddings.py
├── src/data/           # tokenizer.py, dataloader.py
├── src/train/          # trainer.py, config.py
├── src/utils/          # debug.py, export.py
├── configs/gpt1_char.yaml
├── deploy/app.py       # Gradio: Generate + Evaluate
├── tests/              # Unit tests
├── data/tinyshakespeare.txt
├── train.py
└── evaluation_metrics.json

````

---

## Usage

```bash
python deploy/app.py
````

* Type a prompt (e.g., `"To be or not to"`)
* The model will generate text character-by-character in Shakespearean style

---

## References

* **Transformer architecture**: Vaswani et al., *Attention Is All You Need* (2017)
* **Positional embeddings in GPT**: Learned embeddings, GPT-2 style
* Educational guidance from: [Karpathy, “Let’s build GPT from scratch”](https://youtu.be/kCc8FmEb1nY)

> All code and implementation are original, reflecting **the design and behavior described**.

---

```



```
