# SCPC 2025 — Gallery VQA

**Multiple-choice VQA on everyday smartphone photos**

- **Event**: 2025 Samsung Collegiate Programming Challenge: AI (Jun–Jul 2025)
- **Task**: Select the correct answer to multiple-choice questions about everyday gallery photos
- **Approach**: BLIP2-FLAN-T5-XL fine-tuned with LoRA + partial 4-bit quantization on a synthetically generated dataset
- **Result**: **4th** on private leaderboard — score 0.8344 ([Leaderboard](https://dacon.io/competitions/official/236500/leaderboard))

<p align="center">
<img width="70%" alt="leaderboard" src="scpcrank.png" />
</p>

<p align="center">
<img width="70%" alt="mockup" src="https://github.com/user-attachments/assets/25e7ed94-b9ee-49df-af7c-dc516fca9e6d" />
</p>

---

## Overview

The system has three phases:

**1. Synthetic dataset generation** — a three-step pipeline produces 1,218 training examples:
- `Qwen/Qwen-1_8B` generates scene prompts across categories (nature, travel, food, casual)
- `dreamlike-art/dreamlike-photoreal-2.0` renders each prompt into a realistic photo
- `llava-hf/llava-1.5-7b-hf` annotates each image with a description, a multiple-choice question, and the correct answer

**2. Fine-tuning** — `Salesforce/blip2-flan-t5-xl` is trained with:
- 8-bit quantization during training to fit within 40 GB VRAM
- LoRA adapters on the Q-Former (`query`, `key`, `value`, `dense`) — only ~0.1% of parameters are trainable

**3. Two-stage inference** — for each test image:
1. Generate a free-form description conditioned on the image + question
2. Predict the answer letter (A/B/C/D) using the description, image, and choices together

At inference the T5 decoder is swapped to a 4-bit quantized version for lower VRAM usage.

| Model | Public score |
|---|---|
| Baseline | 0.30486 |
| FLAN-T5 only | 0.81298 |
| **This work** | **0.83262** |

---

## Project Structure

```
.
├── configs/
│   └── config.py            # All hyperparameters and paths in one dataclass
├── dataset/
│   ├── generate/
│   │   ├── prompts.py       # Step 1: scene prompt generation (Qwen)
│   │   ├── images.py        # Step 2: image synthesis (Stable Diffusion)
│   │   └── qa_pairs.py      # Step 3: VQA annotation (LLaVA)
│   └── loader.py            # Dataset preprocessing for training
├── model/
│   ├── build.py             # Model loading (training / inference variants)
│   ├── trainer.py           # CustomTrainer + TrainingArguments factory
│   └── predictor.py         # Two-stage inference logic
├── utils/
│   └── postprocess.py       # Answer extraction, submission builder
├── generate_dataset.py      # Entry point: run full generation pipeline
├── train.py                 # Entry point: fine-tune the model
├── inference.py             # Entry point: run inference, save submission
├── pyproject.toml
└── requirements.txt
```

Legacy prototype notebooks and scripts are preserved in `.legacy/`.

---

## Environment

- OS: Linux
- GPU: 1× NVIDIA A100 SXM4 (40 GB VRAM)
- CUDA: 12.8
- Python: 3.10+

---

## Installation

**Step 1 — PyTorch with CUDA 12.8:**

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

**Step 2 — remaining dependencies:**

```bash
pip install -r requirements.txt
# or as an editable install (includes Jupyter extras):
pip install -e ".[jupyter]"
```

---

## Usage

### 1. Generate the training dataset

```bash
python generate_dataset.py
```

Runs the three-step pipeline sequentially. Each step is skipped automatically if its output already exists.

### 2. Fine-tune

```bash
python train.py
```

Saves the LoRA adapter and tokenizer to `./model/finetuned-blip2-flan-t5-xl/`.

### 3. Inference

```bash
python inference.py
```

Reads `./data/given/test.csv`, runs two-stage prediction, and writes `test_inference_final.csv`.

---

## Configuration

All paths, model IDs, and hyperparameters live in `configs/config.py`. Edit the dataclass fields directly — no CLI flags or YAML files needed.

Key fields:

| Field | Default | Description |
|---|---|---|
| `base_model_id` | `Salesforce/blip2-flan-t5-xl` | Base BLIP2 model |
| `lora_r` / `lora_alpha` | 32 / 64 | LoRA rank and scaling |
| `num_epochs` | 5 | Training epochs |
| `batch_size` | 8 | Per-device batch size |
| `input_max_length` | 384 | Tokenizer input truncation |
| `num_prompt_generations` | 300 | Scene prompt generation iterations |
