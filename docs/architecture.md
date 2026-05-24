# Architecture

## Model: BLIP2-FLAN-T5-XL with LoRA

시스템은 [BLIP2](https://arxiv.org/abs/2301.12597) (`Salesforce/blip2-flan-t5-xl`) 위에 구축되며, 세 가지 component로 구성된다:

```
Image
  │
  ▼
ViT (Vision Encoder)
  │  frozen at all times
  ▼
Q-Former (Query Transformer)   ← LoRA adapters injected here
  │
  ▼
FLAN-T5-XL (Language Model)   ← 8-bit (training) or 4-bit NF4 (inference)
  │
  ▼
Answer token
```

### Vision Encoder

ViT가 image feature를 추출한다. Training과 inference 전반에 걸쳐 동결 상태를 유지한다.

### Q-Former

Q-Former는 image feature에 cross-attention을 수행하는 학습 가능한 query token을 사용하여 vision과 language modality를 연결한다.

LoRA adapter는 네 개의 attention weight matrix(`query`, `key`, `value`, `dense`)에 주입된다.  
전체 parameter의 ~0.1%만 학습 가능하다.

### Language Model (FLAN-T5-XL)

Sequence-to-sequence T5 변형 model이 output text를 생성한다.

| Phase | Quantization | Notes |
|-------|-------------|-------|
| Training | 8-bit (`load_in_8bit=True`) | 40 GB VRAM 내에서 동작 |
| Inference | 4-bit NF4 + double quant | T5만 해당; VRAM ~18 GB로 감소 |

---

## Training objective

Model은 target sequence에 대한 표준 **language modeling (cross-entropy) loss**로 학습된다:

```
Description: {description}
Answer: {answer_letter}
```

Label의 padding token과 out-of-vocabulary token은 loss에 기여하지 않도록 `-100`으로 masking된다.

---

## Inference strategy

### Why two stages?

Model에 직접 정답 문자를 출력하도록 prompt하면 hallucination이나 잘못된 형식의 답변이 자주 나온다.  
먼저 description을 생성하면 answer selection step 전에 language model에 더 풍부한 context를 제공하여 신뢰성이 향상된다.

### Stage 1 — Describe

```
image + "Based on the image and question, write a description." → description
```

`max_new_tokens=128`, greedy decoding.

### Stage 2 — Select

```
image + description + question + choices → "A" / "B" / "C" / "D"
```

`max_new_tokens=3`, greedy decoding.  
출력은 `re.search(r"\b([A-D])\b", text)`로 파싱된다.

---

## Comparison with baseline

| System | Public score |
|--------|-------------|
| Baseline (provided) | 0.30486 |
| FLAN-T5 without two-stage | 0.81298 |
| **BLIP2 + LoRA + two-stage** | **0.83262** |

Private leaderboard: **0.8344**, rank **4th**.

```{image} scpcrank.png
:alt: leaderboard
:width: 70%
:align: center
```
