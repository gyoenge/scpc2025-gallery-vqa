# Inference

## Run

```bash
python inference.py
```

`data/given/test.csv`를 읽어 각 행에 대해 정답 문자를 예측하고, `test_inference_final.csv`에 저장한다.

## Two-stage prediction (`model/predictor.py`)

각 test 예제는 두 번의 forward pass로 처리된다.

### Stage 1 — Description generation

**Model에 전달되는 prompt (이미지 포함):**
```
USER: Based on the image and question, write a description.
Question: {question}

Description:
ASSISTANT:
```

질문을 조건으로 이미지에 대한 자유 형식 description을 생성한다 (`max_new_tokens=128`, greedy decoding).

### Stage 2 — Answer selection

**Model에 전달되는 prompt (이미지 + Stage 1 description 포함):**
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

최대 3개의 token을 생성한다 (`max_new_tokens=3`, greedy decoding).  
출력은 `utils/postprocess.py:extract_answer_letter`로 파싱되어 정규식으로 첫 번째 `[A-D]` token을 반환한다. 매칭되지 않으면 `?`가 기본값으로 반환된다.

### Input encoding

모든 input은 `Blip2Processor`로 인코딩된다. Float32 tensor는 device로 이동하기 전에 float16으로 캐스팅된다.

## Model loading (`model/build.py` → `load_blip2_for_inference`)

Inference 시 VRAM을 줄이기 위해 **T5 decoder만** 4-bit quantization으로 교체된다:

| Component | Quantization |
|-----------|-------------|
| BLIP2 vision encoder + Q-Former | float16 |
| T5 language model decoder | 4-bit NF4 + double quant |
| LoRA adapter | float16 base 위에 로드 |

`google/flan-t5-xl`을 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`로 로드한 뒤 `model.language_model`에 할당하고, `PeftModel.from_pretrained`로 LoRA weight를 로드한다.

## Output format

`test_inference_final.csv`는 `sample_submission.csv`의 복사본에 `answer` column이 채워진 형태다:

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
