from configs.config import Config
from dataset.generate.eval_images import download_eval_images
from dataset.generate.qa_pairs import _load_llava, _annotate_images


def main():
    cfg = Config()
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)

    image_dir = cfg.eval_dir / "images"
    existing = list(image_dir.glob("*.jpg")) if image_dir.exists() else []
    if len(existing) < cfg.num_eval_images:
        print("Step 1/2: Downloading eval images (Flickr30k)...")
        download_eval_images(cfg)
    else:
        print(f"Step 1/2: Skipping (found {len(existing)} images)")

    eval_csv = cfg.eval_dir / "eval_question_answer.csv"
    if not eval_csv.exists():
        print("Step 2/2: Annotating with LLaVA...")
        model, processor = _load_llava(cfg)
        _annotate_images(
            model, processor,
            image_dir=cfg.eval_dir / "images",
            output_path=eval_csv,
            id_prefix="EVAL",
        )
    else:
        print(f"Step 2/2: Skipping (found {eval_csv})")

    print(f"Eval dataset ready at {eval_csv}")


if __name__ == "__main__":
    main()
