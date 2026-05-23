import csv

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from configs.config import Config


def generate_images(cfg: Config) -> None:
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.image_model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    input_csv = cfg.generated_dir / "scene_prompt.csv"
    output_dir = cfg.generated_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_csv, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Generating images"):
            scene_id = row["id"]
            scene_prompt = row["generated_text"].strip()
            prompt = (
                f"A photorealistic, candid moment of '{scene_prompt}', "
                "taken with a **smartphone camera**. "
                "The scene should be vibrant and lifelike, capturing the essence of everyday life. "
                "Realistic lighting, natural colors, soft focus, high detail."
            )
            try:
                image = pipe(prompt).images[0]
                image_path = output_dir / f"scene_{scene_id}.jpg"
                image.save(image_path)
            except Exception as e:
                print(f"Error generating image for ID {scene_id}: {e}")
