# Quickstart

세 가지 명령어로 clean environment에서 submission file까지 완성한다.

## 1. Generate the training dataset

```bash
python generate_dataset.py
```

세 단계를 순서대로 실행한다. 각 단계는 output file이 이미 존재하면 건너뛴다:

- **Step 1** — scene prompt 생성 → `data/generated/scene_prompt.csv`
- **Step 2** — 이미지 렌더링 → `data/generated/images/scene_*.jpg`
- **Step 3** — VQA pair annotation → `data/generated/question_answer.csv`

A100 기준 예상 소요 시간: 총 ~3시간 (image synthesis가 bottleneck).

## 2. Fine-tune

```bash
python train.py
```

LoRA adapter와 tokenizer를 `./model/finetuned-blip2-flan-t5-xl/`에 저장한다.

예상 소요 시간: 1,218개 예제, 5 epoch 기준 ~20분.

## 3. Run inference

```bash
python inference.py
```

`./data/given/test.csv`를 읽어 각 행에 대해 two-stage prediction을 수행하고, 결과를 `./test_inference_final.csv`에 저장한다.

## Required file layout

Inference 실행 전, competition data를 `data/given/`에 배치한다:

```
data/given/
├── test.csv
├── sample_submission.csv
└── <test.csv에서 참조하는 이미지들>
```

`test.csv`에는 다음 column이 있어야 한다: `img_path`, `Question`, `A`, `B`, `C`, `D`.
