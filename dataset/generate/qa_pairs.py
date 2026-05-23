import os

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from configs.config import Config

_PROMPT = (
    "<image>\n"
    "USER: Based on the image, write a description and create a multiple-choice question "
    "with four options (A, B, C, D).\n"
    "Answer the question by selecting the best option from A, B, C, or D.\n"
    "Respond only with a single letter: A, B, C, or D.\n"
    "Follow this exact format:\n\n"
    "Description: [detailed description of the image]\n\n"
    "Question: [a question about the image or its content]\n"
    "A. [option A]\n"
    "B. [option B]\n"
    "C. [option C]\n"
    "D. [option D]\n\n"
    "Answer: [A/B/C/D]\n"
    "\n"
    "ASSISTANT:"
)


def generate_qa_pairs(cfg: Config) -> None:
    processor = AutoProcessor.from_pretrained(cfg.qa_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.qa_model_id, torch_dtype=torch.float16, device_map="auto"
    )

    image_dir = cfg.generated_dir / "images"
    output_path = cfg.generated_dir / "question_answer.csv"

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))
    )
    results = []

    for idx, image_file in enumerate(image_files):
        image_path = image_dir / image_file
        image = Image.open(image_path).convert("RGB")
        print(f"Processing {idx + 1}/{len(image_files)}: {image_file}")

        inputs = processor(text=_PROMPT, images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        decoded = processor.decode(output[0], skip_special_tokens=True)
        decoded = decoded[len(_PROMPT) - 6:]

        try:
            desc = decoded.split("Description:")[1].split("Question:")[0].strip()
            question_block = decoded.split("Question:")[1].split("\n")
            question = question_block[0].strip()
            choices = {"A": "", "B": "", "C": "", "D": ""}
            for line in question_block[1:]:
                line = line.strip()
                for letter in ("A", "B", "C", "D"):
                    if line.startswith(f"{letter}."):
                        choices[letter] = line[2:].strip()
            answer = decoded.split("Answer:")[-1].strip()[:1]
        except Exception as e:
            print(f"Parse error in {image_file}: {e}")
            continue

        results.append({
            "ID": f"TRAIN_{idx:03d}",
            "img_path": str(image_path),
            "Description": desc,
            "Question": question,
            "A": choices["A"],
            "B": choices["B"],
            "C": choices["C"],
            "D": choices["D"],
            "answer": answer,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} QA pairs to {output_path}")
