# Architecture

## Model: BLIP2-FLAN-T5-XL with LoRA

The system is built on [BLIP2](https://arxiv.org/abs/2301.12597) (`Salesforce/blip2-flan-t5-xl`), which consists of three components:

```
Image
  │
  ▼
ViT (Vision Encoder)
  │  frozen at all times
  ▼
Q-Former (Query Transformer)   ← LoRA adapters injected here
  │
  ▼
FLAN-T5-XL (Language Model)   ← 8-bit (training) or 4-bit NF4 (inference)
  │
  ▼
Answer token
```

### Vision Encoder

A ViT extracts image features. It is frozen throughout training and inference.

### Q-Former

The Q-Former bridges the vision and language modalities using a set of learnable query tokens that cross-attend to the image features.

LoRA adapters are injected into four attention weight matrices: `query`, `key`, `value`, `dense`.  
Only ~0.1% of total parameters are trainable.

### Language Model (FLAN-T5-XL)

A sequence-to-sequence T5 variant generates the output text.

| Phase | Quantization | Notes |
|-------|-------------|-------|
| Training | 8-bit (`load_in_8bit=True`) | Fits within 40 GB VRAM |
| Inference | 4-bit NF4 + double quant | T5 only; reduces VRAM to ~18 GB |

---

## Training objective

The model is trained with a standard **language modeling (cross-entropy) loss** over the target sequence:

```
Description: {description}
Answer: {answer_letter}
```

Padding tokens and out-of-vocabulary tokens in the labels are masked with `-100` so they do not contribute to the loss.

---

## Inference strategy

### Why two stages?

Directly prompting the model to output a letter often produces hallucinated or malformed answers.  
Generating a description first gives the language model a richer context before the answer selection step, improving reliability.

### Stage 1 — Describe

```
image + "Based on the image and question, write a description." → description
```

`max_new_tokens=128`, greedy decoding.

### Stage 2 — Select

```
image + description + question + choices → "A" / "B" / "C" / "D"
```

`max_new_tokens=3`, greedy decoding.  
The output is parsed by `re.search(r"\b([A-D])\b", text)`.

---

## Comparison with baseline

| System | Public score |
|--------|-------------|
| Baseline (provided) | 0.30486 |
| FLAN-T5 without two-stage | 0.81298 |
| **BLIP2 + LoRA + two-stage** | **0.83262** |

Private leaderboard: **0.8344**, rank **4th**.

```{image} scpcrank.png
:alt: leaderboard
:width: 70%
:align: center
```
