from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl

from configs.config import Config

_GRAD_CHECK_STEP = 5  # warn if grad_norm is still 0 after this many steps


class _GradNormCheckCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or state.global_step > _GRAD_CHECK_STEP:
            return
        grad_norm = logs.get("grad_norm", None)
        if grad_norm is not None:
            status = "OK" if grad_norm > 0 else "ZERO — gradients not flowing"
            print(f"[GradCheck] step={state.global_step} grad_norm={grad_norm:.6f} [{status}]")
            if grad_norm == 0 and state.global_step == _GRAD_CHECK_STEP:
                print("[GradCheck] WARNING: grad_norm has been 0 for all logged steps. "
                      "LoRA may not be receiving gradients.")


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
        fp16=False,
        overwrite_output_dir=True,
    )
    return CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        callbacks=[_GradNormCheckCallback()],
    )
