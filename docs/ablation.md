# Ablation Study

## Run

```bash
python ablation.py --eval_csv <labeled_eval.csv>
python ablation.py --eval_csv <labeled_eval.csv> --output results.csv
```

Eval CSV는 다음 column을 모두 포함해야 한다: `img_path`, `Question`, `A`, `B`, `C`, `D`, `answer`.

## Metrics

각 variant에 대해 세 가지 metric을 계산한다:

| Metric | 설명 |
|--------|------|
| Accuracy | 전체 정답률 |
| Per-class accuracy | A / B / C / D 각 class별 정답률 |
| Confusion matrix | GT(행) × Pred(열) 빈도 행렬 |

## Variants

Variant 목록은 `ablation.py` 상단의 `VARIANTS` 리스트에서 직접 편집한다.

### 기본 제공 variants

| Variant | Inference mode | Model |
|---------|---------------|-------|
| `two_stage + finetuned` | Two-stage | Fine-tuned (LoRA) |
| `single_stage + finetuned` | Single-stage | Fine-tuned (LoRA) |
| `two_stage + base` | Two-stage | Base (no LoRA) |

**Two-stage**: 이미지를 먼저 설명한 뒤 정답 문자를 선택한다 (`model/predictor.py:predict`).  
**Single-stage**: Description 생성 없이 바로 정답 문자를 선택한다 (`model/predictor.py:predict_single_stage`).  
**Base**: LoRA adapter를 로드하지 않은 base BLIP2 model로 inference한다.

### Dataset composition ablation

Training data 구성이 다른 model checkpoint를 비교한다.  
각 조건으로 학습한 뒤 `VARIANTS`에 추가하면 된다:

```python
AblationVariant("two_stage + synthetic_only", trained_model_id="./model/finetuned-synthetic-only"),
AblationVariant("two_stage + synthetic+real", trained_model_id="./model/finetuned-synthetic-real"),
```

### LoRA config ablation

LoRA rank 등 hyperparameter가 다른 checkpoint를 비교한다:

```python
AblationVariant("two_stage + lora_r8",  trained_model_id="./model/finetuned-lora-r8"),
AblationVariant("two_stage + lora_r16", trained_model_id="./model/finetuned-lora-r16"),
AblationVariant("two_stage + lora_r64", trained_model_id="./model/finetuned-lora-r64"),
```

## Output

실행이 끝나면 각 variant의 confusion matrix와 전체 비교 summary table이 출력된다.

```
============================================================
Summary
============================================================
                           accuracy  acc_A   acc_B   acc_C   acc_D  n_invalid
variant
two_stage   + finetuned    0.8326    0.8512  0.8201  0.8344  0.8247  0
single_stage + finetuned   0.7914    ...
two_stage   + base         0.3102    ...
```

`--output`을 지정하면 summary가 CSV로도 저장된다.

## Adding a new variant

`ablation.py`의 `VARIANTS` 리스트에 `AblationVariant`를 추가하면 된다:

```python
AblationVariant(
    name="my_experiment",
    inference_mode="two_stage",       # "two_stage" | "single_stage"
    use_finetuned=True,               # False이면 base model 사용
    trained_model_id="./model/...",   # None이면 cfg.trained_model_id 사용
)
```
