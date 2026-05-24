# Installation

## Requirements

| Component | Version |
|-----------|---------|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA A100 SXM4 40 GB (minimum ~20 GB recommended) |
| CUDA | 12.8 |
| Python | 3.10+ |

## Step 1 — Install PyTorch

PyTorch must be installed separately with the correct CUDA index URL.

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

## Step 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

Or install as an editable package (adds Jupyter extras):

```bash
pip install -e ".[jupyter]"
```

## Verifying the installation

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.version.cuda)          # 12.8
```

## HuggingFace models

The following models are downloaded automatically on first run from HuggingFace Hub:

| Step | Model | Size (approx.) |
|------|-------|---------------|
| Prompt generation | `Qwen/Qwen-1_8B` | 3.5 GB |
| Image synthesis | `dreamlike-art/dreamlike-photoreal-2.0` | 2.1 GB |
| QA annotation | `llava-hf/llava-1.5-7b-hf` | 14 GB |
| Fine-tuning / inference | `Salesforce/blip2-flan-t5-xl` | 15 GB |
| 4-bit decoder (inference) | `google/flan-t5-xl` | 3 GB |

Ensure sufficient disk space (~50 GB) and a HuggingFace account if any model requires gated access.
