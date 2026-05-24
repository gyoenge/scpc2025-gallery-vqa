# Configuration

All hyperparameters and paths are in a single dataclass: `configs/config.py`.

Edit the fields directly — there are no CLI flags, environment variables, or YAML files.

```python
from configs.config import Config

cfg = Config()
cfg.num_epochs = 10   # override a field
```

## Model IDs

| Field | Default | Description |
|-------|---------|-------------|
| `base_model_id` | `Salesforce/blip2-flan-t5-xl` | Base BLIP2 model for training and inference |
| `t5_model_id` | `google/flan-t5-xl` | T5 decoder loaded separately in 4-bit for inference |
| `prompt_model_id` | `Qwen/Qwen-1_8B` | Text-generation model for scene prompt creation |
| `image_model_id` | `dreamlike-art/dreamlike-photoreal-2.0` | Diffusion model for image synthesis |
| `qa_model_id` | `llava-hf/llava-1.5-7b-hf` | Multimodal model for VQA annotation |
| `trained_model_id` | `./model/finetuned-blip2-flan-t5-xl` | Path to the saved LoRA adapter |

## Paths

| Field | Default | Description |
|-------|---------|-------------|
| `generated_dir` | `./data/generated` | Output directory for synthetic generated data |
| `given_dir` | `./data/given` | Directory containing competition test data |
| `real_dir` | `./data/real` | Root directory for real image data (COCO) |
| `output_model_dir` | `./model/finetuned-blip2-flan-t5-xl` | Where `trainer.save_model()` writes |
| `submission_save_path` | `./test_inference_final.csv` | Final submission file |

## Dataset generation

| Field | Default | Description |
|-------|---------|-------------|
| `num_prompt_generations` | 300 | Number of Qwen calls (≈5 scenes each → ~1,500 prompts) |
| `categories` | 4 strings | Scene categories sampled randomly per Qwen call |
| `use_real_data` | `True` | Mix COCO val2017 real images into training data |
| `num_real_images` | 3000 | Number of COCO images to download when `use_real_data=True` |
| `balance_answer_dist` | `False` | Undersample to equalize A/B/C/D counts before training |

## LoRA

| Field | Default | Description |
|-------|---------|-------------|
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | LoRA scaling factor |
| `lora_dropout` | 0.1 | Dropout applied to LoRA weights |
| `lora_target_modules` | `["query", "key", "value", "dense"]` | Q-Former layers to inject adapters into |

## Training

| Field | Default | Description |
|-------|---------|-------------|
| `batch_size` | 8 | Per-device train batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = `batch_size × steps` = 32 |
| `num_epochs` | 5 | Total training epochs |
| `learning_rate` | 5e-5 | AdamW learning rate |
| `logging_steps` | 10 | Log every N optimizer steps |
| `save_steps` | 200 | Save checkpoint every N optimizer steps |
| `save_total_limit` | 3 | Maximum checkpoints to keep on disk |
| `input_max_length` | 384 | Tokenizer input truncation length |
| `target_max_length` | 128 | Tokenizer target truncation length |
