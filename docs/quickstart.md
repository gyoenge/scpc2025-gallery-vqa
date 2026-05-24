# Quickstart

Three commands to go from a clean environment to a submission file.

## 1. Generate the training dataset

```bash
python generate_dataset.py
```

Runs three steps sequentially. Each step is skipped if its output file already exists:

- **Step 1** — generate scene prompts → `dataset/generated/scene_prompt.csv`
- **Step 2** — render images → `dataset/generated/images/scene_*.jpg`
- **Step 3** — annotate with VQA pairs → `dataset/generated/question_answer.csv`

Expected runtime on an A100: ~3 hours total (image synthesis is the bottleneck).

## 2. Fine-tune

```bash
python train.py
```

Saves the LoRA adapter and tokenizer to `./model/finetuned-blip2-flan-t5-xl/`.

Expected runtime: ~20 minutes for 5 epochs on 1,218 examples.

## 3. Run inference

```bash
python inference.py
```

Reads `./dataset/given/test.csv`, runs two-stage prediction on each row, and writes the submission to `./test_inference_final.csv`.

## Required file layout

Before running inference, place the competition data in `dataset/given/`:

```
dataset/given/
├── test.csv
├── sample_submission.csv
└── <images referenced by test.csv>
```

The `test.csv` must have columns: `img_path`, `Question`, `A`, `B`, `C`, `D`.
