"""Download a subset of COCO val2017 images and generate VQA annotations."""

import io
import json
import random
import urllib.request
import zipfile

from tqdm import tqdm

from configs.config import Config
from dataset.generate.qa_pairs import _load_llava, _annotate_images

_COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
_COCO_IMAGE_BASE = "http://images.cocodataset.org/val2017"


def download_coco_images(cfg: Config) -> None:
    image_dir = cfg.real_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    existing = list(image_dir.glob("*.jpg"))
    if len(existing) >= cfg.num_real_images:
        print(f"Skipping download: {len(existing)} real images already exist.")
        return

    ids = _get_coco_val_ids(cfg)
    print(f"Downloading {len(ids)} COCO val2017 images...")
    for img_id in tqdm(ids, desc="COCO images"):
        dest = image_dir / f"{img_id:012d}.jpg"
        if dest.exists():
            continue
        try:
            urllib.request.urlretrieve(f"{_COCO_IMAGE_BASE}/{img_id:012d}.jpg", dest)
        except Exception as e:
            print(f"  Failed {img_id}: {e}")


def generate_real_qa_pairs(cfg: Config) -> None:
    model, processor = _load_llava(cfg)
    _annotate_images(
        model, processor,
        image_dir=cfg.real_dir / "images",
        output_path=cfg.real_dir / "real_question_answer.csv",
        id_prefix="REAL",
    )


def _get_coco_val_ids(cfg: Config) -> list[int]:
    """Return a random subset of COCO val2017 image IDs.

    Downloads annotations once (~240 MB) and caches the ID list locally.
    """
    cache = cfg.real_dir / "coco_val_ids.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        with open(cache) as f:
            all_ids = json.load(f)
    else:
        print("Fetching COCO val2017 image list (one-time ~240 MB download)...")
        with urllib.request.urlopen(_COCO_ANNOTATIONS_URL) as resp:
            data = io.BytesIO(resp.read())
        with zipfile.ZipFile(data) as z:
            with z.open("annotations/instances_val2017.json") as f:
                meta = json.load(f)
        all_ids = [img["id"] for img in meta["images"]]
        with open(cache, "w") as f:
            json.dump(all_ids, f)

    random.seed(42)
    return random.sample(all_ids, min(cfg.num_real_images, len(all_ids)))
