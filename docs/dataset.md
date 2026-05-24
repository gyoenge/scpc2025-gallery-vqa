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

**Output**: `dataset/generated/scene_prompt.csv`

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

**Output**: `dataset/generated/images/scene_<id>.jpg`

---

## Step 3 — QA pair annotation (`dataset/generate/qa_pairs.py`)

**Model**: `llava-hf/llava-1.5-7b-hf` (multimodal)

Each generated image is passed to LLaVA with a structured prompt that instructs the model to produce:

- A description of the image
- A multiple-choice question with four options (A–D)
- The correct answer letter

Responses that fail parsing are silently skipped.

**Output**: `dataset/generated/question_answer.csv`

| Column | Description |
|--------|-------------|
| `ID` | `TRAIN_000`, `TRAIN_001`, … |
| `img_path` | Absolute path to the generated image |
| `Description` | Free-text image description |
| `Question` | The multiple-choice question |
| `A` / `B` / `C` / `D` | Answer choices |
| `answer` | Correct answer letter (`A`–`D`) |

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

To substitute your own labeled data, place a CSV at `dataset/generated/question_answer.csv` with the columns listed above and set `img_path` to absolute or relative paths that `PIL.Image.open` can resolve.
