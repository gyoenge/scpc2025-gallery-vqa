from datasets import Dataset
from transformers import Trainer, TrainingArguments

from configs.config import Config


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def make_trainer(model, dataset: Dataset, cfg: Config, processor) -> CustomTrainer:
    args = TrainingArguments(
        output_dir=str(cfg.output_model_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        save_total_limit=cfg.save_total_limit,
        fp16=True,
    )
    return CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )
