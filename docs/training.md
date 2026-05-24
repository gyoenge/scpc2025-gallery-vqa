# Training

## Run

```bash
python train.py
```

The script loads the model, builds the dataset, trains, and saves in sequence.

## What happens

### 1. Model loading (`model/build.py` → `load_blip2_for_training`)

- **Base model**: `Salesforce/blip2-flan-t5-xl` loaded in **8-bit** via `bitsandbytes`
- `prepare_model_for_kbit_training` freezes non-LoRA weights and enables gradient checkpointing
- LoRA adapters are injected into the Q-Former attention layers (`query`, `key`, `value`, `dense`)

LoRA configuration:

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 32 |
| Alpha | 64 |
| Dropout | 0.1 |
| Target modules | `query`, `key`, `value`, `dense` |
| Task type | `SEQ_2_SEQ_LM` |
| Trainable params | ~0.1% of total |

### 2. Dataset preprocessing (`dataset/loader.py` → `build_dataset`)

CSV 로드 및 병합 후 `dataset/validate.py:validate_qa` 가 실행됩니다:

1. 유효하지 않은 정답 제거 (`A`/`B`/`C`/`D` 이외)
2. 중복 질문 제거
3. A/B/C/D 분포 출력 (15% 미만 클래스에 경고)
4. `balance_answer_dist = True` 이면 undersample로 균형 조정

그 뒤 각 예제를 `(prompt, target)` 쌍으로 변환합니다:

**Prompt** (passed to BLIP2 with the image):
```
USER: Based on the image, write a description and create a multiple-choice question
with four options (A, B, C, D).
Answer the question by selecting the best option from A, B, C, or D.
...
Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Description:
Answer:

ASSISTANT:
```

**Target** (supervised label):
```
Description: {description}
Answer: {answer_letter}
```

Tokenization uses `padding="max_length"` with `input_max_length=384` and `target_max_length=128`.  
Label tokens that are padding (0) or out-of-vocabulary are masked with `-100` so they are excluded from the loss.

### 3. Training (`model/trainer.py`)

- **Trainer**: `CustomTrainer` (subclass of HuggingFace `Trainer`) — strips the `num_items_in_batch` key that newer versions inject
- **Precision**: `fp16=True`
- **Effective batch size**: `batch_size × gradient_accumulation_steps` = 8 × 4 = **32**

Default hyperparameters:

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Epochs | 5 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Logging steps | 10 |
| Save steps | 200 |
| Max checkpoints | 3 |

### 4. Output

The LoRA adapter and tokenizer are saved to `./model/finetuned-blip2-flan-t5-xl/`.

```
model/finetuned-blip2-flan-t5-xl/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

## Customizing training

All parameters live in `Config` (`configs/config.py`). Edit the dataclass directly — no CLI flags or YAML files.

```python
# configs/config.py
@dataclass
class Config:
    lora_r: int = 32          # increase for more capacity
    lora_alpha: int = 64
    num_epochs: int = 5
    learning_rate: float = 5e-5
    batch_size: int = 8
```
