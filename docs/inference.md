# Inference

## Run

```bash
python inference.py
```

Reads `data/given/test.csv`, predicts an answer letter for each row, and writes `test_inference_final.csv`.

## Two-stage prediction (`model/predictor.py`)

Each test example is processed in two forward passes.

### Stage 1 — Description generation

**Prompt sent to the model (with image):**
```
USER: Based on the image and question, write a description.
Question: {question}

Description:
ASSISTANT:
```

The model generates a free-form description of the image conditioned on the question (`max_new_tokens=128`, greedy decoding).

### Stage 2 — Answer selection

**Prompt sent to the model (with image + stage-1 description):**
```
USER: Based on the image, description, and question,
choose the best option from A, B, C, or D.
Description: {description}
Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Answer:
```

The model generates up to 3 tokens (`max_new_tokens=3`, greedy decoding).  
The output is parsed by `utils/postprocess.py:extract_answer_letter` — the first `[A-D]` token found by regex is returned. If none matches, the answer defaults to `?`.

### Input encoding

All inputs are encoded via `Blip2Processor`. Float32 tensors are cast to float16 before moving to the device.

## Model loading (`model/build.py` → `load_blip2_for_inference`)

To reduce VRAM at inference, only the **T5 decoder** is swapped to 4-bit quantization:

| Component | Quantization |
|-----------|-------------|
| BLIP2 vision encoder + Q-Former | float16 |
| T5 language model decoder | 4-bit NF4 + double quant |
| LoRA adapter | loaded on top of float16 base |

This is achieved by loading `google/flan-t5-xl` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` and then assigning it as `model.language_model` before loading the LoRA weights with `PeftModel.from_pretrained`.

## Output format

`test_inference_final.csv` is a copy of `sample_submission.csv` with the `answer` column filled:

| ID | answer |
|----|--------|
| TEST_000 | B |
| TEST_001 | A |
| … | … |

## VRAM budget

| Stage | VRAM used (approx.) |
|-------|-------------------|
| Model loading (inference) | ~18 GB |
| Per-example forward pass | ~20 GB peak |
