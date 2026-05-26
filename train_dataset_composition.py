"""
Dataset composition ablation: trains two checkpoints for direct comparison.

  synthetic_only  —  only AI-generated images + LLaVA QA pairs
  synthetic_real  —  synthetic + COCO val2017 real images

Checkpoints are saved under:
  ./model/finetuned-synthetic-only
  ./model/finetuned-synthetic-real

Usage:
    python train_composition_ablation.py                     # trains both
    python train_composition_ablation.py --composition synthetic_only
    python train_composition_ablation.py --composition synthetic_real
"""

import argparse
import dataclasses
from pathlib import Path

from configs.config import Config
from dataset.loader import build_dataset
from model.build import load_blip2_for_training
from model.trainer import make_trainer


COMPOSITIONS: dict[str, dict] = {
    "synthetic_only": {
        "use_real_data": False,
        "output_model_dir": Path("./model/finetuned-synthetic-only"),
    },
    "synthetic_real": {
        "use_real_data": True,
        "output_model_dir": Path("./model/finetuned-synthetic-real"),
    },
}


def _train_one(name: str, overrides: dict) -> None:
    print(f"\n{'='*60}\nComposition: {name}\n{'='*60}")
    cfg = dataclasses.replace(Config(), **overrides)

    print("Loading model...")
    model, processor = load_blip2_for_training(cfg)

    print("Building dataset...")
    dataset = build_dataset(cfg, processor)

    print("Training...")
    trainer = make_trainer(model, dataset, cfg, processor)
    trainer.train()

    print("Saving...")
    trainer.save_model()
    processor.tokenizer.save_pretrained(str(cfg.output_model_dir))
    print(f"Saved → {cfg.output_model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--composition",
        choices=list(COMPOSITIONS),
        default=None,
        help="Which composition to train. Omit to train both sequentially.",
    )
    args = parser.parse_args()

    targets = [args.composition] if args.composition else list(COMPOSITIONS)
    for name in targets:
        _train_one(name, COMPOSITIONS[name])


if __name__ == "__main__":
    main()
