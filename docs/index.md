# SCPC2025 Gallery VQA

Multimodal Visual Question Answering on smartphone gallery photos.  
Built for the [DACON 2025 Samsung Collegiate Programming Challenge](https://dacon.io/competitions/official/236500/leaderboard).

**Final result: 4th — private score 0.8344**

```{image} scpcrank.png
:alt: leaderboard
:width: 70%
:align: center
```

---

## What this project does

Given a smartphone photo and a multiple-choice question, the system selects the correct answer (A / B / C / D).

The pipeline has three parts:

1. **Synthetic dataset generation** — 1,218 labeled VQA examples generated entirely by AI models
2. **Fine-tuning** — BLIP2-FLAN-T5-XL with LoRA adapters, trained with 8-bit quantization
3. **Two-stage inference** — first describe the image, then select the answer letter

---

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

dataset
training
inference
```

```{toctree}
:maxdepth: 2
:caption: Reference

configuration
architecture
```
