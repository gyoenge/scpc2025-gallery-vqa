import torch
from PIL import Image
from transformers import Blip2Processor

from utils.postprocess import extract_answer_letter


class Predictor:
    def __init__(self, model, processor: Blip2Processor, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device

    def predict(self, image: Image.Image, row) -> str:
        description = self._describe(image, row["Question"])
        return self._select_answer(image, row, description)

    def _describe(self, image: Image.Image, question: str) -> str:
        prompt = (
            "USER: Based on the image and question, write a description.\n"
            f"Question: {question}\n\n"
            "Description:\n"
            "ASSISTANT:"
        )
        inputs = self._encode(image, prompt)
        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _select_answer(self, image: Image.Image, row, description: str) -> str:
        prompt = (
            "USER: Based on the image, description, and question, "
            "choose the best option from A, B, C, or D.\n"
            f"Description: {description}\n"
            f"Question: {row['Question']}\n"
            f"A. {row['A']}\n"
            f"B. {row['B']}\n"
            f"C. {row['C']}\n"
            f"D. {row['D']}\n\n"
            "Answer:"
        )
        inputs = self._encode(image, prompt)
        output = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        decoded = self.processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return extract_answer_letter(decoded)

    def predict_single_stage(self, image: Image.Image, row) -> str:
        prompt = (
            "USER: Based on the image and question, "
            "choose the best option from A, B, C, or D.\n"
            f"Question: {row['Question']}\n"
            f"A. {row['A']}\n"
            f"B. {row['B']}\n"
            f"C. {row['C']}\n"
            f"D. {row['D']}\n\n"
            "Answer:"
        )
        inputs = self._encode(image, prompt)
        output = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        decoded = self.processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return extract_answer_letter(decoded)

    def _encode(self, image: Image.Image, text: str) -> dict:
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        return {
            k: (v.half().to(self.device) if v.dtype == torch.float32 else v.to(self.device))
            for k, v in inputs.items()
        }
