"""Download Flickr30k images for eval dataset.

Flickr30k is sourced from HuggingFace (nlphuji/flickr30k).
Set HF_TOKEN in the environment if the dataset requires authentication.
"""

from tqdm import tqdm

from configs.config import Config


def download_eval_images(cfg: Config) -> None:
    output_dir = cfg.eval_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.jpg"))
    if len(existing) >= cfg.num_eval_images:
        print(f"Skipping download: {len(existing)} eval images already exist.")
        return

    from datasets import load_dataset

    print("Loading Flickr30k (test split) from HuggingFace...")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    n = min(cfg.num_eval_images, len(ds))

    print(f"Saving {n} images to {output_dir}...")
    for i in tqdm(range(n), desc="Eval images"):
        dest = output_dir / f"flickr_{i:04d}.jpg"
        if dest.exists():
            continue
        ds[i]["image"].save(dest)

    print(f"Downloaded {n} eval images to {output_dir}")
