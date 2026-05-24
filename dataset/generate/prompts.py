import csv
import re
import time
import random
import warnings

import torch
from transformers import pipeline
from tqdm import tqdm

from configs.config import Config


def generate_prompts(cfg: Config) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=cfg.prompt_model_id,
        trust_remote_code=True,
        device=device,
    )
    print(f"Prompt model loaded on {'GPU' if device == 0 else 'CPU'}.")

    output_path = cfg.generated_dir / "scene_prompt.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "generated_text"])

        scene_id = 0
        for i in tqdm(range(cfg.num_prompt_generations), desc="Generating scenes (5/gen)", unit="5scene"):
            try:
                category = random.choice(cfg.categories)
                prompt = _build_prompt(category)
                output = generator(
                    prompt, max_new_tokens=512, do_sample=True, temperature=0.9
                )
                result = output[0].get("generated_text") if output else None
                if not result:
                    continue
                result = result[len(prompt):]
                scenes = re.findall(r'\d+\.\s+(.*)', result)[:10]
                for scene in scenes:
                    writer.writerow([scene_id, scene.strip()])
                    scene_id += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"Error at iteration {i}: {e}")

    print(f"Saved {scene_id} scene prompts to {output_path}")


def _build_prompt(category: str) -> str:
    return (
        "Generate a list of 5 distinct and realistic smartphone photo gallery scenes.\n"
        "Describe the scene with following category:\n"
        f"{category}"
        "\n"
        "Strict rules:\n"
        "- Each item must describe a **unique** scene.\n"
        "- **No repetition** of similar phrases or situations.\n"
        "- Result fomat should be like:\n"
        "1. ... \n"
        "2. ... \n"
        "3. ... \n"
        "4. ... \n"
        "5. ... \n"
        "- Each scene should be **detailed** and **vividly** described.\n"
        "\n"
    )
