# GPT-Mini

A minimal GPT trained from scratch on Shakespeare.  
Architecture: GPT-2 style with **learned positional embeddings** and **pre-LayerNorm**.  
~1.23M parameters.  

> **Training Update:** Training steps were increased from 2,000 → **10,000**, and evaluation interval from 500 → **2,500 steps** to improve model performance.

---

## Performance Comparison

| Evaluation Date | Max Steps | Eval Interval | Perplexity | Character Accuracy |
|-----------------|-----------|---------------|------------|------------------|
| Oct 23, 2025    | 2,000     | 500           | 4.40       | 58.7%            |
| Oct 24, 2025    | 10,000    | 2,500         | 3.02       | 64.92%           |

---



