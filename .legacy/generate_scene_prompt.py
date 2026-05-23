from transformers import pipeline
import warnings
import torch 
import csv
import time
import re
from tqdm import tqdm
import random

# Load model 
warnings.filterwarnings("ignore", category=FutureWarning)
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation", 
    model="Qwen/Qwen-1_8B", 
    trust_remote_code=True,
    device=device
)
print(f"Model loaded successfully on {'GPU' if device == 0 else 'CPU'}.")

##### Need To Set ##################
categories = [
    "- Nature (e.g. landscape, animal, weather, plants)\n",
    "- Travel (e.g. tourist spots, local streets, vehicles, airports)\n",
    "- Casual (e.g. daily life, work, family, kids, friends, school, sports)\n",
    "- Food (e.g. meals, cafes, snacks, fruits, drinks)\n",
]
def generate_prompt(category):
    prompt = (
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
    return prompt
output_path = "./dataset/generated/scene_prompt.csv"
num_generations = 300
####################################

# generate and save 
with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "generated_text"])  

    scene_id = 0
    for i in tqdm(range(num_generations), desc="Generating scenes (5scene/1gen)", unit="5scene"):
        try:
            prompt = generate_prompt(random.choice(categories))
            # print(prompt)
            result = generator(prompt, max_new_tokens=2000, do_sample=True, temperature=0.9)[0]["generated_text"]
            result = result[len(prompt):] # result.removeprefix(prompt)
            # print(result)
            scenes = re.findall(r'\d+\.\s+(.*)', result)
            scenes = scenes[:10]
            # print(scenes)
            for scene in scenes:
                writer.writerow([scene_id, str(scene.strip())])
                scene_id += 1
            # print(f"Generated {i + 1}/{num_samples}")
            time.sleep(0.5)  
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
