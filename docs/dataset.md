# Dataset Generation

Training set은 AI 생성 데이터로만 구성된다.  
사람의 annotation이 필요하지 않다.

## Pipeline overview

```
Qwen-1.8B          →  scene_prompt.csv
     ↓
dreamlike-photoreal →  images/scene_*.jpg
     ↓
LLaVA-1.5-7B       →  question_answer.csv
```

전체 pipeline 실행:

```bash
python generate_dataset.py
```

각 단계는 output file이 이미 있으면 건너뛰므로, 중단 후 안전하게 재시작할 수 있다.

실제 COCO 이미지를 혼합하려면 실행 전 `configs/config.py`에서 활성화한다:

```python
use_real_data: bool = True   # Step 4 활성화
num_real_images: int = 500   # 다운로드할 COCO val2017 이미지 수
```

---

## Step 1 — Scene prompt generation (`dataset/generate/prompts.py`)

**Model**: HuggingFace `pipeline("text-generation")`을 통한 `Qwen/Qwen-1_8B`

호출당 5개의 스마트폰 사진 scene description을 생성하도록 prompt된다.  
`Config.num_prompt_generations` (default 300) 호출 × 유효 scene ~4개 ≈ 1,200개 이상의 scene prompt.

매 iteration마다 무작위로 샘플링되는 category:

| Category | Example |
|----------|---------|
| Nature | 풍경, 동물, 날씨, 식물 |
| Travel | 관광지, 골목길, 공항 |
| Casual | 일상생활, 가족, 스포츠, 학교 |
| Food | 식사, 카페, 간식, 음료 |

**Output**: `data/generated/scene_prompt.csv`

| Column | 설명 |
|--------|------|
| `id` | 순차 정수 |
| `generated_text` | 한 문장의 scene description |

---

## Step 2 — Image synthesis (`dataset/generate/images.py`)

**Model**: `dreamlike-art/dreamlike-photoreal-2.0` (Stable Diffusion 변형)

각 scene prompt는 diffusion model에 전달되기 전 사실적 묘사 지시문으로 감싸진다:

```
A photorealistic, candid moment of '{scene}', taken with a smartphone camera.
Realistic lighting, natural colors, soft focus, high detail.
```

**Output**: `data/generated/images/scene_<id>.jpg`

---

## Step 3 — QA pair annotation (`dataset/generate/qa_pairs.py`)

**Model**: `llava-hf/llava-1.5-7b-hf` (multimodal)

각 생성 이미지는 다음을 생성하도록 구조화된 prompt와 함께 LLaVA에 전달된다:

- 이미지 description
- 네 가지 선택지(A–D)가 있는 객관식 질문
- 정답 문자

Parsing에 실패한 response는 자동으로 건너뛴다.

**Output**: `data/generated/question_answer.csv`

| Column | 설명 |
|--------|------|
| `ID` | `TRAIN_000`, `TRAIN_001`, … |
| `img_path` | 생성된 이미지의 절대 경로 |
| `Description` | 자유 형식 이미지 description |
| `Question` | 객관식 질문 |
| `A` / `B` / `C` / `D` | 선택지 |
| `answer` | 정답 문자 (`A`–`D`) |

---

## Step 4 (optional) — Real image augmentation (`dataset/generate/real_qa.py`)

**Source**: COCO val2017 (5,000개의 실제 스마트폰 스타일 사진)

`configs/config.py`에서 `use_real_data = True`로 설정하면 pipeline이:

1. `annotations_trainval2017.zip`을 한 번 다운로드(~240 MB)하여 image ID를 가져오고, `data/real/coco_val_ids.json`에 cache
2. `http://images.cocodataset.org/val2017/`에서 `num_real_images`개의 JPEG 다운로드
3. 동일한 LLaVA annotation step 실행 → `data/real/real_question_answer.csv`

`use_real_data = True`일 때 `dataset/loader.py`가 `question_answer.csv`(synthetic)와 `real_question_answer.csv`(real)를 자동으로 합친다.

| Output | 설명 |
|--------|------|
| `data/real/images/` | 다운로드된 COCO val2017 JPEG |
| `data/real/coco_val_ids.json` | Cache된 image ID 목록 (재다운로드 불필요) |
| `data/real/real_question_answer.csv` | 실제 이미지에 대한 VQA annotation |

---

## QA quality validation (`dataset/validate.py`)

`build_dataset` 호출 직전에 세 단계가 순서대로 실행된다.

### Step 1 — Invalid answer filter

`answer` column이 `A` / `B` / `C` / `D` 이외인 행을 제거한다.  
LLaVA parsing 실패로 빈 값이나 `?`가 들어간 행이 해당된다.

### Step 2 — Duplicate question filter

`Question` column 기준으로 중복 행을 제거한다 (첫 번째 행 유지).

### Step 3 — Answer distribution report

A/B/C/D 비율을 출력하고, 한 class가 15% 미만이면 경고를 표시한다.

```
Answer distribution — A: 1734 (25.0%) | B: 1721 (24.8%) | C: 1743 (25.1%) | D: 1738 (25.1%)
```

불균형이 심한 경우:

```
Answer distribution — A: 891 (12.8%) ⚠ | B: 2341 (33.7%) | ...
```

`balance_answer_dist = True`로 설정하면 class별 최솟값으로 undersample해 균형을 맞춘다.

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

직접 레이블링한 데이터를 사용하려면, 위에 나열된 column을 포함한 CSV를 `data/generated/question_answer.csv`에 배치하고, `PIL.Image.open`이 접근할 수 있는 절대 또는 상대 경로로 `img_path`를 설정하면 된다.
