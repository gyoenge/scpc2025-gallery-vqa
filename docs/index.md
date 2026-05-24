# SCPC2025 Gallery VQA

스마트폰 갤러리 사진을 대상으로 한 멀티모달 Visual Question Answering 시스템.  
[DACON 2025 삼성 대학생 프로그래밍 챌린지](https://dacon.io/competitions/official/236500/leaderboard)를 위해 제작됐다.

**최종 결과: 4위 — Private Score 0.8344**

```{image} scpcrank.png
:alt: leaderboard
:width: 70%
:align: center
```

---

## What this project does

스마트폰 사진과 객관식 질문이 주어지면 정답(A / B / C / D)을 선택한다.

Pipeline은 세 단계로 구성된다:

1. **Synthetic dataset generation** — AI 모델이 완전 자동으로 생성한 레이블된 VQA 예제
2. **Fine-tuning** — LoRA adapter를 적용한 BLIP2-FLAN-T5-XL, 8-bit quantization으로 학습
3. **Two-stage inference** — 이미지를 먼저 설명한 뒤 정답 문자를 선택

---

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

dataset
training
inference
```

```{toctree}
:maxdepth: 2
:caption: Reference

configuration
architecture
```
