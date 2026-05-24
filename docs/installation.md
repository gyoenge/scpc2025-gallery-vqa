# Installation

## Requirements

| 항목 | 버전 |
|------|------|
| OS | Linux (Ubuntu 22.04 테스트) |
| GPU | NVIDIA A100 SXM4 40 GB (최소 ~20 GB 권장) |
| CUDA | 12.8 |
| Python | 3.10+ |

## Step 1 — Install PyTorch

PyTorch는 CUDA index URL을 지정하여 별도로 설치해야 한다.

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

## Step 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

Jupyter extras를 포함한 editable package로 설치하려면:

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

아래 모델들은 최초 실행 시 HuggingFace Hub에서 자동으로 다운로드된다:

| Step | Model | Size (approx.) |
|------|-------|----------------|
| Prompt generation | `Qwen/Qwen-1_8B` | 3.5 GB |
| Image synthesis | `dreamlike-art/dreamlike-photoreal-2.0` | 2.1 GB |
| QA annotation | `llava-hf/llava-1.5-7b-hf` | 14 GB |
| Fine-tuning / inference | `Salesforce/blip2-flan-t5-xl` | 15 GB |
| 4-bit decoder (inference) | `google/flan-t5-xl` | 3 GB |

충분한 disk space(~50 GB)를 확보해야 하며, gated model의 경우 HuggingFace 계정이 필요하다.
