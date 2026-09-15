# 방법론과 평가 기준

목표는 AI가 작성한 한국어 글을 입력받아 **내용을 유지하면서 사람이 쓴 글처럼 재작성하는 generator**를 개발하는 것이다.
MASH의 inverse data construction, style-injection SFT, preference alignment를 한국어에 적용했다.
Detector 회피를 학습 신호와 평가 proxy로 사용하고, 별도 detector와 사람 평가로 효과의 범위를 확인했다.

## 연구 질문

| 질문 | 검증 내용 |
|---|---|
| RQ1 | Generator는 입력의 내용을 보존하면서 detector를 회피할 수 있는가? |
| RQ2 | 한 도메인에서 학습한 generator의 회피 효과가 다른 도메인에서도 유지되는가? |
| RQ3 | 더 강력해진 detector를 학습 신호로 활용하면 generator의 회피 성능이 더 향상되는가? |
| RQ4 | Detector를 회피하도록 학습한 글이 사람에게도 더 사람이 쓴 글처럼 인식되는가? |

여기서 RQ3의 “더 강력해진”은 기존 generator 출력에 적응한 detector를 뜻한다.

## 1. Human–AI pair data 구축

인간 원문을 LLM이 같은 내용으로 재서술하게 하여 `(x_ai, x_human)` pair를 만든다.
생성에 사용한 모델은 Qwen3-8B와 EXAONE-3.5-7.8B이며, 한국어 공용 P3b 프롬프트로 문장 재구성을 요청한다.
사람이 AI 글을 직접 고쳐 대량의 정답을 작성하는 비용을 줄이는 inverse construction 방식이다.
원자료, 뉴스 20,343쌍의 분할 및 실험별 데이터 범위는 [데이터 문서](data.md)를 따른다.

## 2. Detector 학습·검증 및 pair 선별

`klue/roberta-base`를 Human/AI 분류기로 fine-tuning한다.
뉴스에서는 750개 기사 ID에서 얻은 인간 원문과 두 LLM의 재서술로 학습하고, 다른 2,250개 기사 ID로 검증했다.
이후 detector를 고정해 인간 원문은 `P(AI) < 0.5`, AI 재서술은 `P(AI) ≥ 0.5`인 pair를 남겼다.
Detector 학습 기사를 제외하고 선별된 pair를 generator train/dev/test로 분할한다.
Generator 학습 중 detector의 파라미터를 함께 갱신하지 않는다.

## 3. Style-injection SFT

Generator의 기반 모델은 약 **124M, 즉 0.124B** 규모의 `gogamza/kobart-base-v2`다.
AI 입력을 encoder에 넣어 토큰별 content representation `CR`을 만든다.
`CR`의 각 토큰 벡터에 학습 가능한 AI 또는 Human style vector를 이어 붙이고,
공유하는 선형 fusion layer를 거쳐 같은 decoder에 전달한다.

```text
x_ai → Encoder → CR ┬→ concat(CR, AI style)    → shared Fusion → Decoder → x_ai
                   └→ concat(CR, Human style) → shared Fusion → Decoder → x_human
```

- AI 경로는 입력 AI 글의 재구성을 학습한다.
- Human 경로는 대응하는 인간 원문을 생성하도록 학습한다.
- 두 경로는 encoder, fusion layer, decoder를 공유하며 style vector만 다르다.
- Teacher forcing을 사용하고, 전체 loss는 `0.5 × AI 재구성 CE + 0.5 × Human 변환 CE`다.

KoBART의 768차원 표현과 style vector를 concat하여 1,536차원으로 만든 뒤 다시 768차원으로 투영한다.
타깃 끝에 EOS를 명시적으로 추가해 종료를 학습한다. 추론에서는 Human 경로만 사용한다.
대표 설정은 3 epochs, batch 8, learning rate `2e-5`, 최대 학습 길이 512 tokens다.
구현은 [stage2_sft.py](../scripts/stage2_sft.py)에 있다.

## 4. DPOP alignment

SFT generator가 아직 AI로 탐지되는 출력을 이용해 preference data를 만든다.
**입력은 원래의 AI 글**이며, SFT 출력은 rejected 후보로 쓰인다.

| 항목 | 구성 |
|---|---|
| Prompt | 원래 AI 입력 `x_ai` |
| Chosen | 대응하는 인간 원문 `x_human` |
| Rejected | SFT 후보 중 고정 detector가 AI로 판단한 출력 |
| Reference | 학습을 고정한 SFT generator 복사본 |
| Policy | SFT에서 시작해 추가 학습하는 generator |

주 설정에서는 입력마다 후보 4개를 생성하고, `P(AI) > 0.5`인 후보 중 가장 높은 점수를 hard negative로 고른다.
해당 후보가 없으면 그 입력을 preference data에서 제외한다.
Detector 점수는 이 **오프라인 후보 선별**에 사용되며, DPOP loss에서 detector로 역전파하지 않는다.

일반 DPO는 reference 대비 chosen과 rejected의 **상대적인 로그우도 차이**를 키운다.
DPOP는 chosen의 로그우도가 reference보다 낮아질 때 penalty를 추가한다.
토큰당 평균 로그우도를 `s`라 하면 구현의 핵심은 다음과 같다.

```text
margin = (s_policy(chosen) - s_ref(chosen))
       - (s_policy(rejected) - s_ref(rejected))
       - λ × max(0, s_ref(chosen) - s_policy(chosen))
loss = -log sigmoid(β × margin)
```

채택 설정은 **길이 정규화, β=2.0, λ=5.0, 1 epoch, learning rate 5e-6, 유효 batch 16**이다.
구현: [후보 구성](../scripts/stage3_build_dpo.py), [DPOP 학습](../scripts/stage3_dpo.py).

## 5. Evaluation

### Detector와 주 지표

주 평가에서 ASR(Attack Success Rate)은 **전체 평가 입력 중 재작성 결과가 detector 임계값을 통과한 비율**이다.
각 detector의 `P(AI)`가 Human 점수의 95백분위 임계값 `τ` 이하이면 통과로 센다.

```text
Raw ASR = count[P(AI | generator output) ≤ τ] / 전체 평가 입력 수
τ = 해당 평가 집합 Human 점수의 95백분위
```

**Pair 선별에는 0.5, 주 평가에는 해당 평가 집합의 Human P95**를 사용한다.

SCRN은 *Siamese Calibrated Reconstruction Network*의 아이디어를 적용한 별도 평가용 detector다.
실제 구현 backbone은 **`monologg/koelectra-base-v3-discriminator`**다.
서로 다른 노이즈를 넣은 두 잠재 표현에서 원래 표현을 복원하고 분류 결과를 일치시키도록 학습한다.
Generator의 pair 선별·preference 학습에는 SCRN을 사용하지 않았다.

### 의미 유사도·붕괴·비교 실험

의미 유사도는 `jhgan/ko-sroberta-multitask`의 masked mean 표현으로 계산한 **원래 AI 입력–재작성 출력 cosine**이다.
최대 **256 tokens**까지만 반영하므로 장문 전체의 사실·수치 보존 검증을 대신하지 않는다.
급성 반복 붕괴는 공백을 제거한 문자 6-gram이 **20회를 초과해 반복**되는 출력이다.
Raw ASR에는 이런 출력도 포함하므로 붕괴율을 함께 보고한다.
주 평가 생성은 beam 4, 입력 최대 512/output 최대 1,024 tokens이며 반복 억제 옵션을 사용하지 않는다.

뉴스 full test에서는 SFT→DPOP에 따라 RoBERTa ASR **72.8→79.6%**, SCRN ASR **63.3→71.3%**였다.
Cosine은 모두 **0.988**, 붕괴율은 **0.3→0.8%**였다. [뉴스 집계](../results/metrics/news_main_results.json)
Prompting baseline은 한국어 humanizer skill을 배치용 프롬프트로 적용한 것으로,
모델 간 **같은 100개 ID**에서 비교했다.
[Baseline 보고서](../results/reports/humanizer_skill_baseline_n100_20260909.html)

교차 도메인 평가는 각 target의 test와 detector를 고정해 6×6 조합을 비교한다.
t-SNE는 뉴스 test 200편의 Human/AI/SFT/DPOP 텍스트에서 얻은 RoBERTa CLS 표현을 2차원으로 나타낸 보조 시각화다.

## 6. Adversarial alignment와 사람 평가

기존 첫 DPOP generator를 G1, 기존 detector를 D0로 둔다.
Train 기사들의 Human/G0/G1 텍스트를 사용해 D1을 학습하고, D1이 고른 G1 hard negative로 G2를 추가 DPOP 학습한다.
G2의 reference는 고정된 G1이며, D1 학습과 G2 학습을 순차적으로 진행한다.
별도 뉴스 test 1,985편에서 D1 기준 ASR은 **3.38→3.93%**, D0 기준은 **93.90→92.64%**였다.
외부 CopyKiller ASR은 **97.03→97.83%**였다. 모든 detector에서 일관된 향상은 확인되지 않았다.
[상세 결과](../adversarial/RESULTS.md)

사람 평가는 50개 기사×4개 버전을 4명이 나누어 작성 주체를 이진 판별했다.
“사람이 작성했다”는 응답은 Human **70%**, G0 **68%**, G1 **64%**, G2 **58%**였다.
작성 주체 판별의 어려움이 관찰됐으며, generator 추가 학습 후 인간 작성 판단율은 증가하지 않았다.
버전별 50건의 소규모 평가이고 각 기사·버전을 한 명이 판단했으므로 모델 간 차이는 탐색적으로 해석한다.
이 설문은 작성 주체를 측정했으며, 자연스러움·사실성은 직접 평정하지 않았다.
[사람 평가 보고서](../results/reports/human_eval_results_share_20260907.md)

## MASH 대비 변경과 한계

영어 BART 파이프라인을 한국어 KoBART와 P3b로 적용하고, vanilla DPO 대신 길이 정규화 DPOP를 채택했다.
약 0.1B의 사전학습된 generator와 **detector 출력 점수만 활용하는 최적화 방식**으로 회피 효과를 확인했다.
Generator 최적화는 detector의 내부 구조·파라미터·gradient에 의존하지 않는다.

MASH의 **Stage 4: inference-time adversarial refinement**는 적용하지 않았다.
원 논문의 Stage 4는 PPL로 교정 우선순위를 정하고 추가 LLM으로 문장을 다듬은 뒤,
detector 회피가 유지되는 수정만 반영하는 후처리다.
평가용 LLM 간 자연스러움 점수 편차가 커 교정 효과를 일관되게 평가할 기준을 확립하지 못했고, Stage 4 도입을 보류했다.
따라서 본 실험은 **추가 LLM 교정 전 generator 출력**을 평가했으며, 회피 성능을 유지하면서 문장 품질을 개선하는 후처리 효과는 검증하지 못했다.
