# 소규모 데이터 튜닝 전략의 일반화 가능성

## 핵심 주장

> **~1,200개의 합성 데이터와 전체 파라미터의 0.1%에 해당하는 Q-Former LoRA 튜닝만으로,
> 훈련 분포를 벗어난 실제 스마트폰 갤러리 이미지에서도 강건한 VQA 성능이 달성된다.**

이 주장을 세 레이어로 뒷받침한다: 구조적 설계 근거, 이 프로젝트의 다중 데이터셋·다중 지표 실험 결과, 기존 연구와의 연결.

---

## Layer 1 — 구조적 근거

소규모 데이터에서 일반화가 가능한 이유는 과적합이 **아키텍처 수준에서 구조적으로 억제**되기 때문이다.

### LoRA의 rank 제약

LoRA는 원래 가중치 행렬 $W$를 고정하고, 저차원 업데이트 $\Delta W = BA$($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$)만 학습한다.

- rank $r = 32$는 원래 차원에 비해 극히 작은 부분공간만 허용 → 학습 가능한 자유도가 제한되어 소규모 데이터에서도 과적합이 억제된다.
- 사전학습 표현이 $W$에 보존된 채로 adaptation만 일어나므로, pretrain 시 습득한 일반 지식이 유지된다.

> Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022

### Q-Former bottleneck

BLIP2의 Q-Former는 32개의 learnable query token이 ViT 출력에 cross-attention을 수행하여 시각 정보를 언어 모달리티로 압축하는 병목 구조다.

- LoRA를 Q-Former의 `query`, `key`, `value`, `dense` 네 행렬에만 적용: **전체 파라미터의 ~0.1%만 학습 가능**.
- ViT는 훈련·추론 전반에서 완전 동결 → 시각 표현 자체는 변하지 않으며, vision-language alignment만 조정된다.
- 즉, 모델이 배워야 하는 것은 "이미지를 어떻게 해석할까"가 아니라 "이미 추출된 시각 특징을 VQA 형식으로 어떻게 연결할까"에 한정된다.

> Li et al., *"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"*, ICML 2023

---

## Layer 2 — 실험적 근거

### 2.1 데이터셋 간 성능 일관성

| 평가 데이터 | 출처 | 훈련 데이터 대비 | 점수 |
|---|---|---|---|
| Competition test set | 실제 스마트폰 갤러리 (미지) | 완전히 다른 분포 | **0.8344** (private LB) |
| Flickr30k eval set | Flickr (real 사진) | 훈련과 출처 다름 | `inference_ablation.py` 결과 |
| COCO val2017 | COCO (real 사진) | 일부 훈련에 포함 | — |

Competition test set과 Flickr30k 모두에서 유사한 정확도가 관측된다면, 특정 데이터셋에 편향된 것이 아닌 **도메인 gap을 넘는 일반화**의 직접 증거가 된다.

### 2.2 Inference strategy 비교

| 조건 | 정확도 | 비고 |
|---|---|---|
| Two-stage + fine-tuned | **0.8326** | description 먼저 생성 후 answer 선택 |
| Single-stage + fine-tuned | 0.7914 | 바로 answer 선택 |
| Two-stage + base (no LoRA) | 0.3102 | fine-tuning 없음 |

- two-stage → single-stage: **−4.1%p**
  - description 생성이 단순한 출력 형식 문제가 아니라, answer selection 전에 language model에 시각적 context를 명시적으로 주입하는 실질적 효과임을 보여준다.
  - Wei et al. (2022)의 chain-of-thought 논리와 동일한 메커니즘: 중간 추론 단계가 최종 답변의 신뢰성을 높인다.

- fine-tuned → base: **−52.2%p**
  - 소규모 데이터라도 LoRA 튜닝이 없으면 성능이 random 수준(25%)에 가깝다.
  - 반대로, 0.1%의 파라미터만 튜닝해도 +52%p 향상 → PEFT가 few-shot setting에서 얼마나 효율적인지를 직접 보여준다.

> Wei et al., *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*, NeurIPS 2022

### 2.3 Per-class 균형

`inference_ablation.py`는 A / B / C / D 각 class별 정확도와 confusion matrix를 출력한다.

- 4개 class에 걸쳐 per-class accuracy가 균등하게 높다면 → 특정 answer letter에 편향된 학습이 아님을 증명.
- 훈련 데이터의 answer distribution이 A:B:C:D ≈ 25:25:25:25로 설계되어 있으므로, 모델이 이를 학습했다면 per-class 성능도 균등해야 한다.

### 2.4 Dataset composition 비교 (pending)

`train_dataset_ablation.py`로 두 조건을 학습한 뒤 `inference_ablation.py`에서 비교한다.

| 조건 | 학습 데이터 | 기대 Flickr30k 정확도 |
|---|---|---|
| `synthetic_only` | AI 합성 이미지 + LLaVA QA (~1,200개) | baseline |
| `synthetic_real` | 합성 + COCO val2017 real (최대 3,000개 추가) | baseline + Δ |

- `synthetic_real` > `synthetic_only` (Flickr30k 기준): real 이미지 혼합이 OOD 일반화를 실질적으로 향상시킨다는 증거.
- 두 조건 간 차이가 작다면: **합성 데이터만으로 이미 충분한 일반화가 달성된다**는 더 강한 주장이 가능하다.

> He et al., *"Is Synthetic Data from Generative Models Ready for Image Recognition?"*, ICLR 2023

---

## Layer 3 — 기존 연구와의 연결

| 주장 포인트 | 근거 논문 |
|---|---|
| LoRA가 소규모 데이터에서도 full fine-tuning 수준의 성능 달성 | Hu et al., *LoRA*, ICLR 2022 |
| Q-Former bottleneck이 제한된 학습으로 adaptation에 최적 | Li et al., *BLIP-2*, ICML 2023 |
| 합성 데이터가 real domain 성능을 보완 | He et al., *Is Synthetic Data Ready?*, ICLR 2023 |
| 중간 추론 단계(description)가 최종 답변 신뢰성 향상 | Wei et al., *Chain-of-Thought*, NeurIPS 2022 |
| PEFT가 full fine-tuning 대비 파라미터 효율적으로 동등한 일반화 | Ding et al., *Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for PLMs*, 2023 |

---

## 논리의 취약점과 보완 방향

### 현재 취약점

**"소규모 데이터이기 때문에 오히려 일반화된다"는 역설**

소규모 데이터 → 과적합 위험 → 일반화 불가? 라는 반론에 대해, LoRA의 구조적 제약이 이를 방지한다는 논리가 필요하다. 이를 실험으로 직접 보이려면:

- **LoRA rank ablation** (r = 8 / 16 / 32 / 64): rank가 높아질수록 Flickr30k 성능이 오히려 떨어지는 패턴이 나오면, r = 32가 소규모 데이터 환경에서의 generalization sweet spot임을 입증할 수 있다.

### 보완 실험 우선순위

1. **Dataset composition ablation** (`train_dataset_ablation.py`) — 이미 준비됨
2. **LoRA rank ablation** — `train_dataset_ablation.py`와 동일한 구조로 확장 가능
3. **Data scale ablation** (optional) — 100 / 500 / 1,200개 subset으로 학습하여 성능 곡선 확인
