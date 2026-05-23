from diffusers import StableDiffusionPipeline
import torch
import os
import csv
from tqdm import tqdm

# Load model 
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 
)
pipe = pipe.to("cuda")

##### Need To Set ##################
input_csv = "./dataset/generated/scene_prompt.csv"
output_dir = "./dataset/generated/images"
####################################

with open(input_csv, mode="r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in tqdm(reader, desc="Generating images"):
        scene_id = row["id"]
        scene_prompt = row["generated_text"].strip()
        prompt = (
            f"A photorealistic, candid moment of \'{scene_prompt}\', taken with a **smartphone camera**. "
            "The scene should be vibrant and lifelike, capturing the essence of everyday life. "
            "Realistic lighting, natural colors, soft focus, high detail."
        )

        try:
            image = pipe(prompt).images[0]
            image_path = os.path.join(output_dir, f"scene_{scene_id}.jpg")
            image.save(image_path)
        except Exception as e:
            print(f"Error generating image for ID {scene_id}: {e}")


