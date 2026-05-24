# Configuration

모든 hyperparameter와 경로는 단일 dataclass인 `configs/config.py`에 정의돼 있다.

CLI flag, 환경 변수, YAML file은 없다. field를 직접 수정하면 된다.

```python
from configs.config import Config

cfg = Config()
cfg.num_epochs = 10   # field override
```

## Model IDs

| Field | Default | 설명 |
|-------|---------|------|
| `base_model_id` | `Salesforce/blip2-flan-t5-xl` | Training 및 inference에 사용되는 base BLIP2 model |
| `t5_model_id` | `google/flan-t5-xl` | Inference 시 4-bit로 별도 로드되는 T5 decoder |
| `prompt_model_id` | `Qwen/Qwen-1_8B` | Scene prompt 생성용 text generation model |
| `image_model_id` | `dreamlike-art/dreamlike-photoreal-2.0` | Image synthesis용 diffusion model |
| `qa_model_id` | `llava-hf/llava-1.5-7b-hf` | VQA annotation용 multimodal model |
| `trained_model_id` | `./model/finetuned-blip2-flan-t5-xl` | 저장된 LoRA adapter 경로 |

## Paths

| Field | Default | 설명 |
|-------|---------|------|
| `generated_dir` | `./data/generated` | Synthetic data output directory |
| `given_dir` | `./data/given` | Competition test data directory |
| `real_dir` | `./data/real` | 실제 이미지 데이터 root directory (COCO) |
| `output_model_dir` | `./model/finetuned-blip2-flan-t5-xl` | `trainer.save_model()` 저장 경로 |
| `submission_save_path` | `./test_inference_final.csv` | 최종 submission file 경로 |

## Dataset generation

| Field | Default | 설명 |
|-------|---------|------|
| `num_prompt_generations` | 1000 | Qwen 호출 횟수 (호출당 ~5개 scene → ~5,000개 prompt) |
| `categories` | 4개 문자열 | Qwen 호출마다 무작위로 샘플링되는 scene category |
| `use_real_data` | `True` | COCO val2017 실제 이미지를 training data에 혼합 |
| `num_real_images` | 3000 | `use_real_data=True`일 때 다운로드할 COCO 이미지 수 |
| `balance_answer_dist` | `False` | Training 전 A/B/C/D 개수를 undersampling으로 균등화 |

## Eval dataset

| Field | Default | 설명 |
|-------|---------|------|
| `eval_dir` | `./data/eval` | Eval 이미지 및 annotation 저장 directory |
| `num_eval_images` | 500 | 다운로드할 Flickr30k 이미지 수 |

## LoRA

| Field | Default | 설명 |
|-------|---------|------|
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | LoRA scaling factor |
| `lora_dropout` | 0.1 | LoRA weight에 적용되는 dropout |
| `lora_target_modules` | `["query", "key", "value", "dense"]` | Adapter를 주입할 Q-Former layer |

## Training

| Field | Default | 설명 |
|-------|---------|------|
| `batch_size` | 8 | Per-device train batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = `batch_size × steps` = 32 |
| `num_epochs` | 5 | 총 training epoch 수 |
| `learning_rate` | 5e-5 | AdamW learning rate |
| `lr_scheduler_type` | `"cosine"` | LR scheduler 종류 (`"cosine"`, `"linear"` 등) |
| `warmup_ratio` | 0.1 | Warmup에 사용할 전체 step 비율 |
| `logging_steps` | 10 | N optimizer step마다 log 출력 |
| `save_steps` | 200 | N optimizer step마다 checkpoint 저장 |
| `save_total_limit` | 3 | Disk에 유지할 최대 checkpoint 수 |
| `input_max_length` | 384 | Tokenizer input 최대 길이 |
| `target_max_length` | 128 | Tokenizer target 최대 길이 |
