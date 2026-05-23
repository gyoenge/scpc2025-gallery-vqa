import os
import warnings

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from configs.config import Config
from model.build import load_blip2_for_inference
from model.predictor import Predictor
from utils.postprocess import build_submission


def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = Config()

    print("Loading model...")
    model, processor = load_blip2_for_inference(cfg)
    predictor = Predictor(model, processor, device)

    test = pd.read_csv(cfg.given_dir / "test.csv")
    results = []

    for i, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open(cfg.given_dir / row["img_path"]).convert("RGB")
        answer = predictor.predict(image, row)
        print(f"[{i + 1}/{len(test)}] Answer: {answer}")
        results.append(answer)

    build_submission(
        results,
        base_csv_path=str(cfg.given_dir / "sample_submission.csv"),
        save_path=cfg.submission_save_path,
    )


if __name__ == "__main__":
    main()
