# 뉴스 도메인 Generator–Detector 반복 학습 파이프라인

## 1. 무엇을 만드는 파이프라인인가

이 프로젝트의 목표는 AI가 작성한 한국어 뉴스의 **내용은 가능한 한 유지하면서 문체를 사람 뉴스에 가깝게 바꾸는 humanizer**를 학습하는 것이다.

파이프라인에는 두 종류의 모델이 등장한다.

- **Generator G:** AI 뉴스를 입력받아 사람 뉴스 문체로 다시 쓰는 모델
- **Detector D:** 입력 문장이 사람 글인지 AI 글인지 판별하는 모델

일반적인 한 번의 학습은 `Detector가 제공하는 신호로 Generator를 학습`하는 데서 끝난다. 이 프로젝트의 반복 경로는 여기서 한 단계 더 나아간다.

1. 기존 Detector를 회피하는 Generator를 만든다.
2. 그 Generator의 출력을 학습한 새 Detector를 만든다.
3. 새 Detector가 잡아내는 실패 사례를 이용해 Generator를 다시 학습한다.

이를 반복하면 개념적으로 다음과 같은 구조가 된다.

```text
D0 → G1 → D1 → G2 → D2 → G3 → ...
```

다만 이번에 실제로 완료하고 평가한 범위는 다음과 같다.

```text
기존 팀 모델 G1, D0 가져오기
        ↓
G1의 출력으로 D1 재학습
        ↓
D1이 잡아낸 G1 실패 사례로 G2 학습
        ↓
G1과 G2를 동일한 test set에서 평가
```

즉, **반복 가능한 구조를 구현했지만 실제 실험은 `G1 → D1 → G2` 한 번의 추가 반복까지 수행했다.** 새 D2와 G3는 아직 학습하지 않았다.

---

## 2. 전체 구조

이 파이프라인은 크게 두 부분으로 나뉜다.

### 2.1 기존 정본 파이프라인: 첫 humanizer 만들기

기존 팀 파이프라인은 사람 뉴스와 AI 뉴스의 pair를 만든 뒤 StyleBART를 SFT와 DPOP로 학습한다.

```text
사람 뉴스 원문
    │
    ├─ Qwen3-8B가 같은 내용을 AI 문체로 재작성 ──> G0
    │
    └─────────────────────────────────────────> Human

(Human, G0) pair
    ↓
StyleBART SFT
    ↓
SFT가 생성했지만 Detector가 여전히 AI로 잡는 문장 수집
    ↓
DPOP 학습
    ↓
첫 humanizer G1
```

여기서 G0는 humanizer가 아니라 **humanizer의 입력이 되는 초기 AI 뉴스**다. G1이 실제로 G0를 사람 문체에 가깝게 다시 쓰는 첫 모델이다.

### 2.2 반복 파이프라인: Detector와 Generator를 한 번씩 갱신

```text
                    ┌──────────────────────┐
                    │ 기존 humanizer G1   │
                    └──────────┬───────────┘
                               │ train 문서 15,780개 생성
                               v
Human + G0 + G1 출력 ───────> 새 Detector D1
                               │
                               │ G1 후보 중 D1이 AI로 잡은 문장 선택
                               v
                    DPOP로 G2 학습
                               │
                               v
                    동일 test ID 1,985개 평가
```

D1과 G2는 동시에 학습하지 않는다. D1을 먼저 학습하고 통과 여부를 확인한 다음 D1을 고정한다. 그 후 D1 점수를 이용해 G2의 preference data를 만든다.

따라서 이 방법은 일반적인 GAN처럼 하나의 계산 그래프에서 Generator와 Discriminator가 동시에 gradient를 주고받는 방식이 아니다. 정확히는 **Generator와 Detector가 번갈아 상대의 출력에 적응하는 순차적 adversarial learning**이다.

---

## 3. 모델과 용어

| 이름 | 의미 | 모델 또는 데이터 |
| --- | --- | --- |
| Human | 실제 사람이 작성한 뉴스 | Generator의 목표 문장 |
| G0 또는 `x_ai` | 초기 AI 뉴스 | Qwen 계열 모델이 사람 뉴스를 다시 쓴 문장 |
| D0 | 기존 팀 뉴스 Detector | frozen 평가 앵커이자 기존 학습 신호 |
| G1 | 기존 팀 humanizer | StyleBART SFT + DPOP checkpoint |
| D1 | G1 출력에 적응한 새 Detector | `klue/roberta-base`를 재초기화해 학습 |
| G2 | D1의 실패 신호로 학습한 새 humanizer | G1을 reference로 한 DPOP 모델 |
| D2, G3 | 다음 반복의 모델 | 이번 실험에서는 생성하지 않음 |

### 팀 checkpoint 이름과 반복 인덱스의 충돌

팀 파일 이름은 다음과 같다.

- `models/news_dpo_D2/dpo_bart.pt`
- `models/news_roberta_D2/`

이 이름의 `D2`는 팀 모델의 버전명이다. 반복 실험의 두 번째 Detector라는 뜻이 아니다. 반복 문맥에서는 이 두 모델을 각각 **G1**, **frozen D0**로 부른다.

---
## 4. 첫 Generator G1이 만들어지는 원리

반복 실험은 완성된 G1에서 시작하지만, G1의 구조를 이해하려면 그 앞의 SFT와 DPOP를 알아야 한다.

### 4.1 Human–AI pair 구축

사람 뉴스 원문을 Qwen3-8B가 같은 내용의 AI 뉴스로 다시 쓴다.

```text
입력: 사람 뉴스 원문
생성: Qwen3-8B, P3b prompt, temperature 0.8
출력: 초기 AI 뉴스 G0
```

그다음 도메인 Detector가 다음 조건을 만족하는 pair만 남긴다.

- Human은 사람 글로 판정되어야 한다.
- G0는 AI 글로 판정되어야 한다.

이 과정을 거치면 Generator가 배워야 할 `(G0, Human)` 쌍이 생긴다.

### 4.2 StyleBART 구조

Generator는 `gogamza/kobart-base-v2`를 기반으로 한 약 124M 규모의 encoder-decoder 모델이다.

일반 BART에 다음 세 요소를 추가한다.

- AI style embedding `ASR`
- Human style embedding `HSR`
- content와 style을 결합하는 shared fusion layer

여기서 style embedding의 `ASR`은 **AI Style Representation**을 뜻한다. 평가 지표의 ASR(Attack Success Rate, 탐지 회피율)과 약어만 같고 서로 다른 개념이다. `HSR`은 Human Style Representation이다.

tensor 흐름은 다음과 같다.

```text
input_ids                         [B, S]
    ↓ BART encoder
content representation CR        [B, S, d]

style embedding                   [d]
    ↓ batch와 sequence 축으로 확장
broadcast style                   [B, S, d]

concat(CR, style)                 [B, S, 2d]
    ↓ Linear(2d → d)
fused representation              [B, S, d]
    ↓ BART decoder
output token logits               [B, T, vocab]
```

fusion layer는 초기 상태에서 content 부분을 그대로 통과시키도록 초기화한다.

```text
fusion([content; style]) ≈ content
```

학습 초기에 style embedding이 content를 갑자기 훼손하지 않도록 한 설계다. 이전 additive 방식은 content representation을 직접 교란해 반복 문장과 출력 붕괴를 일으켰기 때문에 concat 후 projection 방식으로 변경했다.

### 4.3 SFT의 이중 경로

같은 G0 입력을 두 경로로 학습한다.

#### AI reconstruction 경로

```text
입력: G0
style: ASR
정답: G0
loss: L_recon
```

입력의 내용을 보존하면서 AI style로 복원하는 경로다.

#### Human style transfer 경로

```text
입력: G0
style: HSR
정답: Human
loss: L_trans
```

같은 content representation을 사람 문체로 변환하는 경로다.

전체 SFT loss는 다음과 같다.

```text
L_SFT = λ_sft · L_recon + (1 - λ_sft) · L_trans
```

정본에서는 `λ_sft=0.5`이므로 내용 복원과 사람 문체 변환을 같은 비중으로 학습한다. 추론할 때는 HSR 경로만 사용한다.

### 4.4 EOS 처리

ko-BART tokenizer가 target 문장 끝에 EOS를 자동으로 붙이지 않기 때문에 label 끝에 EOS token을 직접 추가한다.

```text
target tokens + EOS
pad 위치만 -100으로 masking
```

EOS도 loss 계산 대상이다. 이 처리가 없으면 모델이 문장의 종료를 학습하지 못해 최대 길이까지 같은 구절을 반복하는 문제가 발생했다.

### 4.5 SFT 학습 설정

| 항목 | 설정 |
| --- | ---: |
| backbone | `gogamza/kobart-base-v2` |
| epochs | 3 |
| batch size | 8 |
| learning rate | `2e-5` |
| SFT `λ` | 0.5 |
| max source/target length | 512 |
| optimizer | AdamW |
| scheduler | OneCycleLR, warm-up 구간 6% |
| gradient clipping | 1.0 |
| fusion | concat 후 `Linear(2d → d)` |
| target label | EOS 포함, pad만 loss 제외 |

### 4.6 SFT hard negative 만들기

SFT가 생성한 문장 중 이미 Detector를 회피한 문장은 학습할 필요가 적다. 반대로 SFT를 통과했는데도 Detector가 AI로 판정한 문장은 Generator의 실패 사례다.

```text
G0 하나에서 여러 human-style 후보 생성
    ↓
Detector로 각 후보의 P(AI) 계산
    ↓
P(AI) > threshold인 후보만 유지
    ↓
가장 AI답다고 판단된 후보를 hard negative로 선택
```

preference pair는 다음과 같다.

```text
prompt   = G0
chosen   = Human
rejected = SFT hard negative
reference policy = frozen SFT
```

이 preference로 DPOP를 학습해 첫 humanizer G1을 만든다.

---

## 5. DPOP를 사용하는 이유

### 5.1 일반 DPO

DPO는 학습 중인 policy가 chosen을 rejected보다 상대적으로 선호하도록 만든다.

```text
h_dpo = [log πθ(yw|x) - log πref(yw|x)]
        - [log πθ(yl|x) - log πref(yl|x)]

L_DPO = -log σ(β · h_dpo)
```

- `x`: G0 입력
- `yw`: Human chosen
- `yl`: Detector가 잡아낸 hard negative
- `πref`: frozen reference Generator
- `πθ`: 학습 중인 Generator

문제는 상대적 차이만 커지면 loss가 작아질 수 있다는 점이다. chosen 확률을 유지하는 대신 rejected 확률만 과도하게 낮추거나, 모델 전체가 reference에서 멀어져 이상한 문장을 만들어도 Detector 점수는 낮아질 수 있다.

실제 실험에서도 vanilla DPO는 반복 문장과 의미 붕괴를 만들면서 Detector만 회피하는 reward hacking을 보였다.

### 5.2 DPOP의 chosen 보존항

DPOP는 chosen의 확률이 reference보다 낮아질 때만 penalty를 추가한다.

```text
h = h_dpo
    - λ_dpop · max(0, log πref(yw|x) - log πθ(yw|x))

L_DPOP = -log σ(β · h)
```

직관적으로는 다음과 같다.

- Human을 hard negative보다 선호하게 만든다.
- 동시에 Human 정답의 likelihood가 원래 Generator보다 낮아지는 것을 막는다.
- Detector를 속이기 위해 문장 자체를 망가뜨리는 방향을 억제한다.

정본은 `λ_dpop=5`를 사용한다. `λ=1`보다 회피 강도는 약하지만 출력 보존을 더 강하게 우선한 설정이다.

### 5.3 DPOP 학습 설정

| 항목 | 설정 |
| --- | ---: |
| epochs | 1 |
| batch size | 2 |
| gradient accumulation | 8 |
| effective batch size | 16 |
| learning rate | `5e-6` |
| `β` | 2.0 |
| DPOP `λ` | 5.0 |
| sequence log-probability | token 평균 length normalization |
| gradient clipping | 1.0 |

length normalization을 켜면 긴 문장이 token 수 때문에 자동으로 불리해지는 현상을 줄일 수 있다. log-probability의 scale이 작아지므로 `β=2.0`을 함께 사용한다.

---

## 6. 기존 팀 모델을 반복 파이프라인으로 가져오기

반복 실험은 팀의 기존 데이터와 checkpoint를 직접 덮어쓰지 않는다. `import_team_news.py`가 원본을 읽기 전용으로 다루고 `loop/runs/team_news` 아래에 별도 형식으로 변환한다.

| 팀 산출물 | 반복 파이프라인에서의 이름 | 변환 결과 |
| --- | --- | --- |
| `dataset/news_track/news_dpair_D2.jsonl` | paired data | gzip JSONL snapshot |
| `models/news_dpo_D2/dpo_bart.pt` | G1 | Hugging Face BART base + style head |
| `models/news_roberta_D2/` | frozen D0 | 원본 directory 복사 + threshold 기록 |

G1 checkpoint 변환에서는 다음 state를 옮긴다.

- `bart.*` tensor → BART base
- `asr`, `hsr` → 두 style embedding
- `fusion.weight`, `fusion.bias` → shared fusion layer

변환 후 source path와 SHA256을 manifest에 기록한다. 데이터 ID 중복도 검사한다.

---

## 7. G1 출력으로 D1 학습하기

### 7.1 G1 replay 생성

train split의 G0 15,780개를 G1의 human style 경로로 변환한다.

```text
G0 train 15,780개
    ↓ G1, greedy generation
G1 train output 15,780개
```

생성 결과는 `doc_id`와 함께 append 방식으로 저장한다. 실행이 중단돼도 기존 결과를 다시 생성하지 않고 이어서 실행할 수 있다.

다음 조건을 모두 검사한다.

- 중복 `doc_id`가 없어야 한다.
- 모든 ID가 현재 train split에 속해야 한다.
- 빈 생성문이 없어야 한다.
- 최종 생성 수가 train 15,780개와 정확히 같아야 한다.
- 생성 결과의 ID 집합과 train ID 집합이 정확히 같아야 한다.

### 7.2 D1 학습 buffer

D1은 세 종류의 문장을 학습한다.

| 데이터 | 개수 | label | sampling 총가중치 |
| --- | ---: | ---: | ---: |
| Human | 15,780 | 0, 사람 | 50% |
| G0 | 15,780 | 1, AI | 25% |
| G1 output | 15,780 | 1, AI | 25% |
| 합계 | 47,340 |  | 100% |

AI class 전체가 50%이고, 그 안에서 기존 AI G0와 새로운 G1 출력을 같은 비중으로 둔다.

G1 출력만 학습하면 Detector가 일반적인 AI 글을 잊고 “ko-BART 출력 탐지기”로 퇴화할 수 있다. G0를 replay에 유지하는 이유가 여기에 있다.

### 7.3 D1 모델과 학습 설정

| 항목 | 설정 |
| --- | ---: |
| backbone | `klue/roberta-base` |
| 초기화 | 기존 D0에서 이어받지 않고 재초기화 |
| max length | 512 |
| epochs | 3 |
| batch size | 8 |
| gradient accumulation | 2 |
| effective batch size | 16 |
| learning rate | `2e-5` |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| optimizer | AdamW |
| random seed | 42 |

### 7.4 threshold calibration과 collapse gate 분리

dev 1,973개는 deterministic하게 두 부분으로 나눈다.

```text
dev-cal  1,022개: threshold τ 결정
dev-gate   951개: D1이 무너지지 않았는지 독립 판정
```

`dev-cal`의 Human score에서 목표 FPR 5%가 되는 지점을 threshold `τ`로 정한다.

그 threshold를 그대로 `dev-gate`에 적용해 다음 조건을 확인한다.

```text
Human FPR ≤ 8%
original-AI TPR ≥ 80%
```

둘 중 하나라도 실패하면 D1은 collapsed로 기록되고 G2 학습을 시작하지 않는다.

calibration과 gate를 같은 문서에서 계산하지 않는 이유는, threshold를 Human FPR 5%에 맞춘 동일 표본에서 다시 FPR을 재면 통과가 사실상 구성적으로 보장되기 때문이다.

---

## 8. D1 신호로 G2 학습하기

### 8.1 G2 runner의 시작 조건

G2 runner는 다음 조건을 먼저 확인한다.

- D1 `gate.json`에서 `collapsed=false`
- D1 Human FPR과 original-AI TPR이 기준 통과
- `gate.json`과 `tau.json`의 threshold 일치
- D1 `model.safetensors` 존재
- G1 `style_head.pt` 존재
- train 문서가 정확히 15,780개
- train과 held-out ID 교집합이 없음
- 기존 실행이 동시에 진행 중이지 않음

하나라도 맞지 않으면 학습을 중단한다.

### 8.2 후보 생성

각 train G0에 대해 G1의 human style 경로로 8개의 후보를 sampling한다.

| 항목 | 설정 |
| --- | ---: |
| 문서당 기본 후보 | 8 |
| 부족할 때 최대 후보 | 16 |
| temperature | 1.0 |
| top-p | 0.95 |
| generation batch | 4 |
| max source/target length | 512 |

beam search 대신 sampling을 쓰는 이유는 같은 입력에서 서로 다른 실패 사례를 확보하기 위해서다.

후보 생성 후에는 모든 train ID가 존재하고 각 ID에 최소 8개 후보가 있는지 확인한다. hard negative가 부족하면 threshold를 낮추지 않고 후보를 최대 16개까지 추가 생성한다.

threshold를 낮추지 않는 이유는 Detector가 충분히 AI라고 판단하지 않은 애매한 후보를 rejected로 사용하면 preference label noise가 늘어나기 때문이다.

### 8.3 preference 구성

각 후보를 D1으로 채점한다.

```text
D1(candidate) > τ
```

조건을 만족한 후보만 hard negative다. 문서당 D1 점수가 높은 순으로 최대 2개를 사용한다.

각 preference는 다음 내용을 가진다.

```text
x_ai       = G0 입력
y_w        = 같은 ID의 Human
y_l        = D1이 AI로 탐지한 G1 후보
d_l        = D1의 P(AI)
ref_logp_w = frozen G1이 Human에 부여한 평균 log-probability
ref_logp_l = frozen G1이 hard negative에 부여한 평균 log-probability
```

runner는 다음을 다시 검증한다.

- preference가 train ID에만 속하는가
- `x_ai`와 `Human`이 원래 pair와 정확히 일치하는가
- chosen과 rejected가 서로 다른가
- rejected의 D1 score가 실제로 `τ`보다 큰가
- detector score와 reference log-probability가 모두 finite인가
- 유효 preference가 최소 하나 이상 존재하는가

### 8.4 G2 DPOP 학습

G1을 복사해 policy G2를 만들고 원본 G1은 frozen reference로 유지한다.

```text
policy    = trainable G2, G1 checkpoint에서 초기화
reference = frozen G1
chosen    = Human
rejected  = D1 hard negative
```

G2 설정은 다음과 같다.

| 항목 | 설정 |
| --- | ---: |
| objective | length-normalized DPOP |
| epochs | 1 |
| batch size | 2 |
| gradient accumulation | 8 |
| effective batch size | 16 |
| learning rate | `5e-6` |
| `β` | 2.0 |
| DPOP `λ` | 5.0 |
| gradient clipping | 1.0 |

학습 후 loss, preference accuracy, margin이 모두 finite인지 확인하고 `style_head.pt` checkpoint가 실제로 생성됐는지 검사한다.

### 8.5 평가 후 의도적으로 종료

G2는 test 1,985개에서 평가한다. runner는 `metrics.json`의 `n_test=1985`를 확인한 뒤 완료 상태를 기록한다.

이 runner는 **D2를 자동으로 재학습하지 않는다.** 새 D2를 만들려면 G2 train output을 별도로 생성하고 새로운 실험으로 preregister해야 한다. Detector gate가 실패했는데도 다음 Generator를 학습하는 일을 막기 위한 경계다.

---

## 9. 파이프라인 산출물

```text
loop/runs/team_news/
├─ import_manifest.json
├─ stage1/
│  └─ dpair.jsonl.gz
├─ stage2/
│  └─ best/                         # imported G1
├─ stage0/
│  └─ detector_d0/                  # frozen D0
└─ arms/team_news/
   ├─ round0/
   │  └─ detector/
   │     ├─ replay_gen.jsonl.gz      # G1 train output
   │     ├─ model/                   # D1 checkpoint
   │     ├─ tau.json
   │     └─ gate.json
   └─ round1/
      ├─ g2_config.yaml
      ├─ g2_status.json
      ├─ candidates.jsonl.gz
      ├─ prefs.jsonl.gz
      ├─ prefs_report.json
      ├─ dpo/
      │  ├─ history.json
      │  └─ final/style_head.pt      # G2 checkpoint
      └─ eval/
         └─ metrics.json
```

원문, 생성문, checkpoint, 외부 평가 PDF는 Git에 넣지 않는다. 코드, config, 집계 결과만 공개 저장소에 포함한다.

---

## 10. 결과

### 10.1 D1 학습과 gate

D1은 3 epoch 중 dev-cal AUC가 가장 높은 epoch 3을 선택했다.

| 항목 | 결과 | 통과 기준 |
| --- | ---: | ---: |
| epoch 3 dev-cal AUC | 0.999686 | 최고 AUC epoch 선택 |
| threshold `τ` | 0.9996544123 | dev-cal Human FPR 목표 5% |
| dev-gate Human FPR | 6.31% (60/951) | ≤ 8% |
| dev-gate original-AI TPR | 100.00% (951/951) | ≥ 80% |
| collapsed | false | false |

동일한 test 1,985개에서 G1을 평가하면 다음과 같다.

| Detector | G1 AI 판정률 | G1 ASR |
| --- | ---: | ---: |
| 기존 D0 | 6.10% | 93.90% |
| 새 D1 | 96.62% | 3.38% |

D0를 잘 회피하던 G1 출력도 D1이 그 패턴을 학습하자 대부분 다시 탐지됐다. 이는 반복 학습이 필요한 이유를 보여준다.

다만 D1이 G1 train output으로 학습됐기 때문에 이 결과는 **in-domain adaptation** 증거다. 새로운 Generator나 다른 도메인에 대한 일반화 증거는 아니다.

### 10.2 G1과 G2의 내부 평가

| 평가 Detector | G1 ASR | G2 ASR | G2−G1 | 해석 |
| --- | ---: | ---: | ---: | --- |
| frozen team D0 | 93.90% | 92.64% | −1.26%p | 고정 앵커에서는 G2 개선 없음 |
| D1 | 3.38% | 3.93% | +0.55%p | 학습 신호와 같은 Detector라 참고용 |
| 새 D2 | 미측정 | 미측정 | — | 아직 학습하지 않음 |

D1 회피율은 G2가 직접 최적화한 Detector에 대한 결과라 순환적이다. 이것만으로 G2가 좋아졌다고 판단할 수 없다.

### 10.3 CopyKiller 전수 paired 평가

학습 과정에 참여하지 않은 상용 Detector CopyKiller로 동일 test ID 1,985개를 평가했다. AI작성률 50% 이상을 AI 판정으로 정의했다.

| 버전 | 평균 AI작성률 | AI 판정률 | ASR |
| --- | ---: | ---: | ---: |
| Human | 1.303% | 0.86% (17/1,985) | — |
| G0 | 39.258% | 40.20% (798/1,985) | 59.80% |
| G1 | 3.913% | 2.97% (59/1,985) | 97.03% |
| G2 | 3.285% | 2.17% (43/1,985) | 97.83% |

G1에서 G2로 갈 때:

- 평균 AI작성률 변화: `−0.628%p`
- bootstrap 95% CI: `−1.084 ~ −0.176%p`
- ASR 변화: `+0.806%p`
- exact McNemar: `p=0.01133`
- 점수 변화 방향 sign-test: `p=0.16346`

G1과 G2 모두 CopyKiller를 매우 높은 비율로 회피했다. 그러나 G2의 추가 개선은 약 0.81%p로 작고 통계 지표에 따라 해석이 달라진다.

또한 content-identical DOCX 700개를 재검사했을 때 692개만 동일한 점수를 받았고, 8개는 최대 46%p 차이가 났다. CopyKiller 결과는 검사 batch와 재검사 변동성이 있는 black-box 측정값으로 해석해야 한다.

### 10.4 블라인드 사람 평가

사람 평가에서는 공통 기사 ID 50개와 Human/G0/G1/G2 네 버전을 사용했다. 평가자 4명이 각각 50개 문서를 보았고, 같은 평가자는 같은 기사 ID의 다른 버전을 보지 않았다.

참가자에게는 모델명, 실제 정답, Detector score를 숨기고 `AI가 작성했다 / 사람이 작성했다` 중 하나를 선택하게 했다.

| 버전 | 사람이 작성했다고 판단한 비율 | 95% Wilson CI |
| --- | ---: | ---: |
| Human | 70% (35/50) | 56.2–80.9% |
| G0 | 68% (34/50) | 54.2–79.2% |
| G1 | 64% (32/50) | 50.1–75.9% |
| G2 | 58% (29/50) | 44.2–70.6% |

세 AI 버전 모두 절반 이상이 사람 글로 판단됐다. 그러나 관측값은 G0, G1, G2 순으로 낮아졌고 신뢰구간이 크게 겹친다.

조건별 표본이 50개이고 각 문서·버전을 한 명만 평가했으므로 G0/G1/G2의 순위를 확정할 수는 없다. 대신 다음 사실을 보여준다.

> CopyKiller 점수가 낮아지는 것과 사람이 더 자연스럽거나 인간답다고 느끼는 것은 같은 목표가 아니다.

### 10.5 내용 보존과 품질 측정

G1에 대해 측정한 결과는 다음과 같다.

| 지표 | 결과 | 의미 |
| --- | ---: | --- |
| Human style feature까지의 z-distance | G0 0.389 → G1 0.190 | 표면 문체 특성이 Human 쪽으로 이동 |
| G1 길이 / Human 길이 | 평균 90.1% | 평균적으로 조금 짧아짐 |
| 고유명사 recall | 평균 85.9% | 일부 고유명사 유실 가능 |
| G0 대비 수치 표현 누락 문서율 | 21.26% | 수치 보존 위험 신호 |
| G0 대비 수치 표현 추가 문서율 | 34.46% | 수치 추가 또는 표기 변화 가능 |

수치 지표는 정규식 기반 휴리스틱이다. `2026년`과 `올해` 같은 표기 변화나 같은 숫자의 출현 횟수 변화도 경고로 잡을 수 있으므로 위 비율을 사실 오류율로 해석하면 안 된다.

전체 G1/G2 실험은 light evaluation으로 실행돼 SimCSE와 PPL을 측정하지 않았다. 따라서 현재 결과는 탐지 회피를 잘 설명하지만, 의미 보존·사실성·자연스러움을 완전히 검증하지 못한다.

---

## 11. 결과를 어떻게 해석해야 하는가

### 확인된 것

1. 기존 G1은 frozen D0를 93.90% 회피했다.
2. G1 output을 학습한 D1은 G1을 다시 96.62% 탐지했다.
3. D1의 hard negative를 이용해 G2까지 학습하는 전체 코드 경로가 정상 완료됐다.
4. G1과 G2는 CopyKiller에서 각각 97.03%, 97.83%의 높은 ASR을 보였다.
5. 블라인드 사람 평가에서도 AI 버전의 58–68%가 사람 글로 판단됐다.

### 확인되지 않은 것

1. G2가 G1보다 전반적으로 더 좋은 글을 쓴다는 결론
2. G2가 새 D2를 더 잘 회피한다는 결론
3. fact distortion이 충분히 낮다는 결론
4. 사람 뉴스로서 자연스러움이 개선됐다는 결론
5. 뉴스에서의 결과가 다른 한국어 도메인으로 전이된다는 결론

핵심 결과는 “반복할수록 계속 좋아졌다”가 아니다.

> 작은 humanizer만으로도 상용 AI Detector의 판정을 크게 바꿀 수 있었지만, Detector 회피의 개선이 사람 관점의 품질 개선으로 그대로 이어지지는 않았다.

---
