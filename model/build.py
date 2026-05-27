import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    T5ForConditionalGeneration,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, TaskType

from configs.config import Config


def load_blip2_for_training(cfg: Config):
    processor = Blip2Processor.from_pretrained(cfg.base_model_id, use_fast=True)

    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        load_in_8bit=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_blip2_base(cfg: Config):
    """Load BLIP2 with 4-bit quantized T5 decoder, without any LoRA adapter."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = Blip2Processor.from_pretrained(cfg.base_model_id, use_fast=True)

    t5 = T5ForConditionalGeneration.from_pretrained(
        cfg.t5_model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.language_model = t5

    return model, processor


def load_blip2_for_inference(cfg: Config):
    """Load BLIP2 with 4-bit quantized T5 decoder and LoRA adapter."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = Blip2Processor.from_pretrained(cfg.base_model_id, use_fast=True)

    # Load only the T5 decoder in 4-bit to save VRAM
    t5 = T5ForConditionalGeneration.from_pretrained(
        cfg.t5_model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.language_model = t5

    model = PeftModel.from_pretrained(model, cfg.trained_model_id)

    return model, processor
