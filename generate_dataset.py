import csv
import time

import pandas as pd

from configs.config import Config
from dataset.generate.prompts import generate_prompts
from dataset.generate.images import generate_images
from dataset.generate.qa_pairs import generate_qa_pairs
from dataset.generate.real_qa import download_coco_images, generate_real_qa_pairs

_SEP = "=" * 60


def _step(n: int, total: int, label: str) -> None:
    print(f"\n{_SEP}")
    print(f"  Step {n}/{total}  |  {label}")
    print(_SEP)


def _skip(reason: str) -> None:
    print(f"  → Skipped : {reason}")


def _done(elapsed: float) -> None:
    print(f"  → Done    ({elapsed:.1f}s)")


def _count_csv(path) -> int:
    with open(path) as f:
        return sum(1 for _ in csv.reader(f)) - 1  # subtract header


def main():
    cfg = Config()
    cfg.generated_dir.mkdir(parents=True, exist_ok=True)

    total = 4 if cfg.use_real_data else 3
    wall_start = time.time()

    # Step 1 — Scene prompt generation
    _step(1, total, "Scene prompt generation  (Qwen)")
    prompt_csv = cfg.generated_dir / "scene_prompt.csv"
    if prompt_csv.exists():
        _skip(f"found {prompt_csv.name}  ({_count_csv(prompt_csv):,} prompts)")
    else:
        t = time.time()
        generate_prompts(cfg)
        _done(time.time() - t)

    # Step 2 — Image synthesis
    _step(2, total, "Image synthesis  (Stable Diffusion)")
    qa_csv = cfg.generated_dir / "question_answer.csv"
    if qa_csv.exists():
        _skip("question_answer.csv already exists — images assumed complete")
    else:
        t = time.time()
        generate_images(cfg)
        _done(time.time() - t)

    # Step 3 — QA annotation
    _step(3, total, "QA pair annotation  (LLaVA)")
    if qa_csv.exists():
        _skip(f"found {qa_csv.name}  ({_count_csv(qa_csv):,} examples)")
    else:
        t = time.time()
        generate_qa_pairs(cfg)
        _done(time.time() - t)

    # Step 4 — Real image augmentation (optional)
    if cfg.use_real_data:
        _step(4, total, "Real image augmentation  (COCO val2017 + LLaVA)")
        real_qa_csv = cfg.real_dir / "real_question_answer.csv"
        if real_qa_csv.exists():
            n = len(pd.read_csv(real_qa_csv))
            _skip(f"found {real_qa_csv.name}  ({n:,} examples)")
        else:
            t = time.time()
            print("  Downloading COCO val2017 images...")
            download_coco_images(cfg)
            print("  Annotating with LLaVA...")
            generate_real_qa_pairs(cfg)
            _done(time.time() - t)

    # Summary
    total_elapsed = time.time() - wall_start
    print(f"\n{_SEP}")
    print(f"  All steps complete  |  total elapsed: {total_elapsed:.1f}s")
    print(_SEP)


if __name__ == "__main__":
    main()
