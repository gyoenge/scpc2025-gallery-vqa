import csv
import re
import random
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from configs.config import Config


def generate_prompts(cfg: Config) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.prompt_model_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.prompt_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Prompt model loaded on {device}.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                attention_mask = torch.ones_like(input_ids)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False,
                    )
                new_tokens = output_ids[0][input_ids.shape[1]:]
                result = tokenizer.decode(new_tokens, skip_special_tokens=True)
                scenes = re.findall(r'\d+\.\s+(.*)', result)[:10]
                for scene in scenes:
                    writer.writerow([scene_id, scene.strip()])
                    scene_id += 1
            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                continue

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
