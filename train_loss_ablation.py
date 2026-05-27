"""
Loss ablation trainer: trains checkpoints with different loss targets.

  full_loss     —  cross-entropy over full target ("Description: ... Answer: X")
  answer_only   —  cross-entropy over answer token only ("Answer: X")

Checkpoints are saved under:
  ./model/finetuned-full-loss
  ./model/finetuned-answer-only

Usage:
    python train_loss_ablation.py                   # trains both sequentially
    python train_loss_ablation.py --loss full_loss
    python train_loss_ablation.py --loss answer_only
"""

import argparse
import dataclasses
from pathlib import Path

from configs.config import Config
from dataset.loader import build_dataset
from model.build import load_blip2_for_training
from model.trainer import make_trainer


LOSS_CONDITIONS: dict[str, dict] = {
    "full_loss": {
        "answer_only_loss": False,
        "output_model_dir": Path("./model/finetuned-full-loss"),
    },
    "answer_only": {
        "answer_only_loss": True,
        "output_model_dir": Path("./model/finetuned-answer-only"),
    },
}


def _train_one(name: str, overrides: dict) -> None:
    print(f"\n{'='*60}\nLoss condition: {name}\n{'='*60}")
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
        "--loss",
        choices=list(LOSS_CONDITIONS),
        default=None,
        help="Which loss condition to train. Omit to train both sequentially.",
    )
    args = parser.parse_args()

    targets = [args.loss] if args.loss else list(LOSS_CONDITIONS)
    for name in targets:
        _train_one(name, LOSS_CONDITIONS[name])


if __name__ == "__main__":
    main()
