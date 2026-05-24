# Dataset Generation

The training set is built entirely from AI-generated data.  
No human annotation is required.

## Pipeline overview

```
Qwen-1.8B          →  scene_prompt.csv
     ↓
dreamlike-photoreal →  images/scene_*.jpg
     ↓
LLaVA-1.5-7B       →  question_answer.csv
```

Run the full pipeline:

```bash
python generate_dataset.py
```

Each step checks for its output file and skips if already present, so the pipeline is safe to resume after interruption.

To mix in real COCO images, enable `use_real_data` in `configs/config.py` before running:

```python
use_real_data: bool = True   # enables Step 4
num_real_images: int = 500   # how many COCO val2017 images to download
```

---

## Step 1 — Scene prompt generation (`dataset/generate/prompts.py`)

**Model**: `Qwen/Qwen-1_8B` via HuggingFace `pipeline("text-generation")`

The model is prompted to produce 5 distinct smartphone-photo scene descriptions per call.  
`Config.num_prompt_generations` (default 300) calls × ~4 valid scenes each ≈ 1,200+ scene prompts.

Categories sampled randomly each iteration:

| Category | Example |
|----------|---------|
| Nature | landscape, animal, weather, plants |
| Travel | tourist spots, local streets, airports |
| Casual | daily life, family, sports, school |
| Food | meals, cafes, snacks, drinks |

**Output**: `data/generated/scene_prompt.csv`

| Column | Description |
|--------|-------------|
| `id` | Sequential integer |
| `generated_text` | One-sentence scene description |

---

## Step 2 — Image synthesis (`dataset/generate/images.py`)

**Model**: `dreamlike-art/dreamlike-photoreal-2.0` (Stable Diffusion variant)

Each scene prompt is wrapped in a photorealism instruction before being passed to the diffusion model:

```
A photorealistic, candid moment of '{scene}', taken with a smartphone camera.
Realistic lighting, natural colors, soft focus, high detail.
```

**Output**: `data/generated/images/scene_<id>.jpg`

---

## Step 3 — QA pair annotation (`dataset/generate/qa_pairs.py`)

**Model**: `llava-hf/llava-1.5-7b-hf` (multimodal)

Each generated image is passed to LLaVA with a structured prompt that instructs the model to produce:

- A description of the image
- A multiple-choice question with four options (A–D)
- The correct answer letter

Responses that fail parsing are silently skipped.

**Output**: `data/generated/question_answer.csv`

| Column | Description |
|--------|-------------|
| `ID` | `TRAIN_000`, `TRAIN_001`, … |
| `img_path` | Absolute path to the generated image |
| `Description` | Free-text image description |
| `Question` | The multiple-choice question |
| `A` / `B` / `C` / `D` | Answer choices |
| `answer` | Correct answer letter (`A`–`D`) |

---

---

## Step 4 (optional) — Real image augmentation (`dataset/generate/real_qa.py`)

**Source**: COCO val2017 (5,000 real smartphone-style photos)

Set `use_real_data = True` in `configs/config.py`. The pipeline:

1. Downloads `annotations_trainval2017.zip` once (~240 MB) to retrieve image IDs, then caches them as `data/real/coco_val_ids.json`
2. Downloads `num_real_images` individual JPEGs from `http://images.cocodataset.org/val2017/`
3. Runs the same LLaVA annotation step → `data/real/real_question_answer.csv`

`dataset/loader.py` automatically concatenates `question_answer.csv` (synthetic) and `real_question_answer.csv` (real) when `use_real_data = True`.

| Output | Description |
|--------|-------------|
| `data/real/images/` | Downloaded COCO val2017 JPEGs |
| `data/real/coco_val_ids.json` | Cached image ID list (no re-download needed) |
| `data/real/real_question_answer.csv` | VQA annotations for real images |

---

## Dataset statistics

| Metric | Value |
|--------|-------|
| Total examples | 1,218 |
| Images | 1,218 JPEGs |
| Categories | 4 |
| Avg. question length | ~12 tokens |
| Answer distribution | Roughly uniform (A–D) |

---

## Using a custom dataset

To substitute your own labeled data, place a CSV at `data/generated/question_answer.csv` with the columns listed above and set `img_path` to absolute or relative paths that `PIL.Image.open` can resolve.
