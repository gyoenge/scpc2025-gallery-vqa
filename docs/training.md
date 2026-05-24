# Training

## Run

```bash
python train.py
```

Model 로드, dataset 빌드, training, 저장을 순서대로 수행한다.

## What happens

### 1. Model loading (`model/build.py` → `load_blip2_for_training`)

- **Base model**: `bitsandbytes`를 통해 **8-bit**로 로드된 `Salesforce/blip2-flan-t5-xl`
- `prepare_model_for_kbit_training`으로 LoRA 외 weight를 동결하고 gradient checkpointing 활성화
- Q-Former attention layer(`query`, `key`, `value`, `dense`)에 LoRA adapter 주입

LoRA 설정:

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 32 |
| Alpha | 64 |
| Dropout | 0.1 |
| Target modules | `query`, `key`, `value`, `dense` |
| Task type | `SEQ_2_SEQ_LM` |
| Trainable params | ~0.1% of total |

### 2. Dataset preprocessing (`dataset/loader.py` → `build_dataset`)

CSV 로드 및 병합 후 `dataset/validate.py:validate_qa`가 실행된다:

1. 유효하지 않은 정답 제거 (`A`/`B`/`C`/`D` 이외)
2. 중복 질문 제거
3. A/B/C/D 분포 출력 (15% 미만 class에 경고)
4. `balance_answer_dist = True`이면 undersample로 균형 조정

그 뒤 각 예제를 `(prompt, target)` 쌍으로 변환한다:

**Prompt** (이미지와 함께 BLIP2에 전달):
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

Tokenization은 `padding="max_length"` 방식으로, `input_max_length=384`, `target_max_length=128`을 사용한다.  
Padding(0) 또는 out-of-vocabulary token은 loss 계산에서 제외되도록 `-100`으로 masking된다.

### 3. Training (`model/trainer.py`)

- **Trainer**: `CustomTrainer` (HuggingFace `Trainer` subclass) — 최신 버전에서 주입되는 `num_items_in_batch` key를 제거
- **Precision**: `fp16=True`
- **Effective batch size**: `batch_size × gradient_accumulation_steps` = 8 × 4 = **32**

Default hyperparameter:

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

LoRA adapter와 tokenizer가 `./model/finetuned-blip2-flan-t5-xl/`에 저장된다.

```
model/finetuned-blip2-flan-t5-xl/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

## Customizing training

모든 parameter는 `Config` (`configs/config.py`)에 있다. CLI flag나 YAML file 없이 dataclass를 직접 수정하면 된다.

```python
# configs/config.py
@dataclass
class Config:
    lora_r: int = 32          # 용량을 늘리려면 값을 높인다
    lora_alpha: int = 64
    num_epochs: int = 5
    learning_rate: float = 5e-5
    batch_size: int = 8
```
