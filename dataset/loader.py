import torch
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import Blip2Processor

from configs.config import Config


def build_dataset(cfg: Config, processor: Blip2Processor) -> Dataset:
    qa_csv = cfg.generated_dir / "question_answer.csv"
    df = pd.read_csv(qa_csv)

    real_csv = cfg.real_dir / "real_question_answer.csv"
    if cfg.use_real_data and real_csv.exists():
        df_real = pd.read_csv(real_csv)
        df = pd.concat([df, df_real], ignore_index=True)
        print(f"Dataset: {len(df)} total ({len(df_real)} real + {len(df) - len(df_real)} synthetic)")
    df["prompt"] = df.apply(_build_prompt, axis=1)
    df["target"] = df.apply(_build_target, axis=1)

    dataset = Dataset.from_pandas(df)
    vocab_size = processor.tokenizer.vocab_size

    def preprocess(example):
        image = Image.open(example["img_path"]).convert("RGB")
        inputs = processor(
            text=example["prompt"],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.input_max_length,
        )
        labels = processor.tokenizer(
            example["target"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.target_max_length,
        ).input_ids
        return {
            "input_ids": inputs["input_ids"][0].to(torch.long),
            "attention_mask": inputs["attention_mask"][0].to(torch.long),
            "pixel_values": inputs["pixel_values"][0].to(torch.float32),
            "labels": torch.tensor(labels[0], dtype=torch.long),
        }

    def sanitize_labels(example):
        example["labels"] = [
            token if 0 <= token < vocab_size else -100
            for token in example["labels"]
        ]
        return example

    def zero_to_ignore(example):
        example["labels"] = [-100 if token == 0 else token for token in example["labels"]]
        return example

    def cast_attention_mask(example):
        if "attention_mask" in example:
            example["attention_mask"] = torch.tensor(example["attention_mask"]).to(torch.float32)
        if "decoder_attention_mask" in example:
            example["decoder_attention_mask"] = torch.tensor(example["decoder_attention_mask"]).to(torch.float32)
        return example

    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    processed = processed.map(sanitize_labels)
    processed = processed.map(zero_to_ignore)
    processed = processed.map(cast_attention_mask)
    return processed


def _build_prompt(row) -> str:
    return (
        "USER: Based on the image, write a description and create a multiple-choice question "
        "with four options (A, B, C, D).\n"
        "Answer the question by selecting the best option from A, B, C, or D.\n"
        "Respond only with a single letter: A, B, C, or D.\n"
        "Follow this exact format:\n\n"
        f"Question: {row['Question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n\n"
        "Description:\n"
        "Answer:\n\n"
        "ASSISTANT:"
    )


def _build_target(row) -> str:
    return f"Description: {row['Description']}\nAnswer: {row['answer']}"
