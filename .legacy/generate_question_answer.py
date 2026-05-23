import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

# Load model 
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

##### Need To Set ##################
image_dir = "./dataset/generated/images"
output_path = "./dataset/generated/question_answer.csv"
####################################

image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))])
results = []

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert("RGB")

    print("##################################################################")
    print(f"| Processing image {idx + 1}/{len(image_files)}: {image_file}")

    prompt = (
        "<image>\n"
        "USER: Based on the image, write a description and create a multiple-choice question with four options (A, B, C, D).\n"
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

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    decoded = decoded[len(prompt)-6:]  # Remove the prompt part from the output

    print(f"Decoded output: {decoded}")

    try:
        desc = decoded.split("Description:")[1].split("Question:")[0].strip()
        question_block = decoded.split("Question:")[1].split("\n")
        question = question_block[0].strip()
        choices = {"A": "", "B": "", "C": "", "D": ""}
        for line in question_block[1:]:
            line = line.strip()
            if line.startswith("A."):
                choices["A"] = line[2:].strip()
            elif line.startswith("B."):
                choices["B"] = line[2:].strip()
            elif line.startswith("C."):
                choices["C"] = line[2:].strip()
            elif line.startswith("D."):
                choices["D"] = line[2:].strip()
        # answer = decoded.split("ASSISTANT:")[-1].strip()[:1]
        answer = decoded.split("Answer:")[-1].strip()[:1]
    except Exception as e:
        print(f"⚠️ Parse error in {image_file}: {e}")
        continue

    results.append({
        "ID": f"TRAIN_{idx:03d}",
        "img_path": image_path,
        "Description": desc,
        "Question": question,
        "A": choices["A"],
        "B": choices["B"],
        "C": choices["C"],
        "D": choices["D"],
        "answer": answer
    })

df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"| 저장 완료: {output_path}")
