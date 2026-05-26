"""
Ablation study runner.

Requires a labeled eval CSV with columns:
    img_path, Question, A, B, C, D, answer

Usage:
    python ablation.py --eval_csv <path>
    python ablation.py --eval_csv <path> --output results.csv
"""

import argparse
import copy
import os
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from configs.config import Config
from model.build import load_blip2_base, load_blip2_for_inference
from model.predictor import Predictor

LABELS = ["A", "B", "C", "D"]


@dataclass
class AblationVariant:
    name: str
    inference_mode: Literal["two_stage", "single_stage"] = "two_stage"
    use_finetuned: bool = True
    trained_model_id: Optional[str] = None  # overrides cfg.trained_model_id when set


# Edit this list to add, remove, or reorder variants.
VARIANTS: list[AblationVariant] = [
    # Inference strategy
    AblationVariant("two_stage   + finetuned", inference_mode="two_stage",    use_finetuned=True),
    AblationVariant("single_stage + finetuned", inference_mode="single_stage", use_finetuned=True),
    # Fine-tuning
    AblationVariant("two_stage   + base",       inference_mode="two_stage",    use_finetuned=False),
    # Dataset composition
    AblationVariant("two_stage + synthetic_only", trained_model_id="./model/finetuned-synthetic-only"),
    AblationVariant("two_stage + synthetic_real",  trained_model_id="./model/finetuned-synthetic-real"),
    # LoRA rank ablation — uncomment and set paths after training:
    # AblationVariant("two_stage + lora_r8",  trained_model_id="./model/finetuned-lora-r8"),
    # AblationVariant("two_stage + lora_r16", trained_model_id="./model/finetuned-lora-r16"),
    # AblationVariant("two_stage + lora_r64", trained_model_id="./model/finetuned-lora-r64"),
]


def _run_variant(
    variant: AblationVariant,
    df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> list[str]:
    cfg_v = copy.copy(cfg)
    if variant.trained_model_id:
        cfg_v.trained_model_id = variant.trained_model_id

    if variant.use_finetuned:
        model, processor = load_blip2_for_inference(cfg_v)
    else:
        model, processor = load_blip2_base(cfg_v)

    predictor = Predictor(model, processor, device)
    preds = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=variant.name):
        image = Image.open(row["img_path"]).convert("RGB")
        if variant.inference_mode == "two_stage":
            pred = predictor.predict(image, row)
        else:
            pred = predictor.predict_single_stage(image, row)
        preds.append(pred)

    del model
    torch.cuda.empty_cache()
    return preds


def _compute_metrics(gt: list[str], preds: list[str]) -> dict:
    n = len(gt)
    accuracy = sum(g == p for g, p in zip(gt, preds) if p in LABELS) / n
    n_invalid = sum(1 for p in preds if p not in LABELS)

    per_class: dict[str, float] = {}
    for label in LABELS:
        idxs = [i for i, g in enumerate(gt) if g == label]
        if not idxs:
            per_class[label] = float("nan")
            continue
        per_class[label] = sum(1 for i in idxs if preds[i] == label) / len(idxs)

    label_idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((4, 4), dtype=int)
    for g, p in zip(gt, preds):
        if g in LABELS and p in LABELS:
            cm[label_idx[g]][label_idx[p]] += 1

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_invalid": n_invalid,
    }


def _print_confusion(name: str, cm: np.ndarray) -> None:
    col_header = "         " + "  ".join(f"pred_{l}" for l in LABELS)
    print(f"\nConfusion matrix — {name}")
    print(col_header)
    for i, label in enumerate(LABELS):
        row_vals = "  ".join(f"{cm[i][j]:6d}" for j in range(4))
        print(f"  gt_{label}   {row_vals}")


def _print_summary(all_metrics: dict[str, dict]) -> None:
    rows = []
    for name, m in all_metrics.items():
        row: dict = {"variant": name, "accuracy": f"{m['accuracy']:.4f}"}
        for l in LABELS:
            v = m["per_class"][l]
            row[f"acc_{l}"] = f"{v:.4f}" if not np.isnan(v) else "-"
        row["n_invalid"] = m["n_invalid"]
        rows.append(row)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(pd.DataFrame(rows).set_index("variant").to_string())


def main() -> None:
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", required=True, help="Labeled eval CSV with 'answer' column")
    parser.add_argument("--output", default=None, help="Optional path to save summary as CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)
    required = ["img_path", "Question", "A", "B", "C", "D", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eval CSV missing columns: {missing}")

    gt = df["answer"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Eval samples: {len(df)} | Variants: {len(VARIANTS)}")

    cfg = Config()
    all_metrics: dict[str, dict] = {}

    for variant in VARIANTS:
        print(f"\n{'='*60}\n{variant.name}\n{'='*60}")
        preds = _run_variant(variant, df, cfg, device)
        m = _compute_metrics(gt, preds)
        all_metrics[variant.name] = m
        print(f"Accuracy: {m['accuracy']:.4f} | Invalid predictions: {m['n_invalid']}")
        _print_confusion(variant.name, m["confusion_matrix"])

    _print_summary(all_metrics)

    if args.output:
        rows = []
        for name, m in all_metrics.items():
            row: dict = {"variant": name, "accuracy": m["accuracy"], "n_invalid": m["n_invalid"]}
            for l in LABELS:
                row[f"acc_{l}"] = m["per_class"][l]
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
