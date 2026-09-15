# 뉴스 adversarial alignment

**기존 generator G1 → 적응한 detector D1 → 추가 학습한 generator G2**의 한 사이클을 실험했다.
더 강력해진 detector의 점수로 preference를 다시 구성하면 회피 성능이 향상되는지 확인했다.

## 모델과 데이터 버전

| 이름 | 역할 |
|---|---|
| Human | 실제 인간 원문, generator의 chosen |
| G0 | Human에서 만든 원래 AI 글, generator의 입력 |
| D0 | 기존 뉴스 detector |
| G1 | 기존 Style-SFT와 첫 DPOP를 마친 generator |
| D1 | Human/G0/G1 train 글로 새로 학습한 detector |
| G2 | D1에 탐지된 G1 후보로 추가 DPOP 학습한 generator |

별도의 뉴스 데이터 **19,738쌍(train 15,780 / dev 1,973 / test 1,985)**을 사용했다.
추가 실험은 greedy decoding, 입력·출력 최대 512 tokens를 사용한다.
주 실험은 20,343쌍 / test 2,044, beam 4·출력 최대 1,024 tokens로 데이터와 decoding 조건이 다르다.

입력 경로는 아래와 같다. 파일명에 쓰인 `D2`는 기존 뉴스 모델의 버전명이다.

```text
dataset/news_track/news_dpair_D2.jsonl
models/news_dpo_D2/dpo_bart.pt      → G1
models/news_roberta_D2/            → D0
```

이 입력들은 공개 저장소에 포함하지 않는다. Import 과정은 원본을 읽고
`adversarial/runs/team_news/`에 별도 형식으로 복사·변환한다.

## 1. G1 출력에 적응한 D1 학습

1. Train 기사들의 AI 입력을 G1으로 재작성한다.
2. Human은 인간, 원래 AI 글 G0와 G1 출력은 AI로 라벨링한다.
3. `klue/roberta-base` 사전학습 가중치에서 시작해 새 detector를 학습한다.

학습 데이터의 가중치는 **Human 50%, G0 25%, G1 25%**다.

Dev는 기사 ID에 따라 **cal 1,022편 / gate 951편**으로 나눈다.
Cal에서 epoch와 Human FPR 5% 기준의 임계값을 정하고, 별도 gate에서 인간 오탐과 원래 AI 탐지 성능을 확인한다.
선택한 D1 임계값은 **0.9996544**, gate의 Human FPR은 **6.31%**, 원래 AI TPR은 **100%**다.

D1의 적응 효과는 같은 뉴스 도메인의 G1 test 출력에서 평가했다.

## 2. D1으로 preference data 구성

G1에 train의 원래 AI 글을 넣어 문서당 후보 8개를 샘플링하고, 필요하면 16개까지 늘린다.
D1 점수가 정한 임계값을 넘는 후보 중 문서당 최대 2개를 rejected로 사용한다.

| 항목 | 내용 |
|---|---|
| Input | 원래 AI 글 G0 |
| Chosen | 대응하는 인간 원문 |
| Rejected | D1에 여전히 AI로 탐지되는 G1 후보 |
| Reference | 고정한 G1 |
| Policy 초기값 | G1의 복사본 |

D1 학습을 마친 뒤 파라미터를 고정하고, 그 점수로 G2 학습용 preference를 구성한다.

## 3. G2 DPOP 학습

토큰당 평균 로그우도를 사용한 DPOP를 한 번 더 적용한다.
Reference 대비 chosen/rejected 선호 차이를 키우고, chosen 로그우도가 reference보다 낮아질 때 penalty를 부여한다.

| 설정 | 값 |
|---|---:|
| β | 2 |
| DPOP λ | 5 |
| Epoch | 1 |
| Learning rate | 5e-6 |
| 유효 batch | 16 |

G1 reference 로그우도는 미리 계산해 저장한다.
학습 loss는 [일반 방법론의 DPOP 설명](methodology.md#4-dpop-alignment)과 같은 형태다.

## 4. 결과와 해석

같은 test 1,985편에서 평가한 결과다.

| Detector | G1 ASR | G2 ASR | G2−G1 |
|---|---:|---:|---:|
| D0 | 93.90% | 92.64% | −1.26%p |
| D1 | 3.38% | 3.93% | +0.55%p |
| CopyKiller | 97.03% | 97.83% | 약 +0.81%p |

CopyKiller 차이는 탐지된 문서 **59편→43편**, 즉 `16/1,985`에서 계산했다.
CopyKiller는 AI 작성률 **50%**를 임계값으로 사용하는 외부 평가이며 학습 신호로 쓰지 않았다.

D1 학습 후 G1 출력의 AI 탐지율은 D0의 **6.10%**에서 D1의 **96.62%**로 높아졌다.
추가 DPOP는 D1과 CopyKiller에서 소폭 개선됐으나 D0에서는 감소했다.
즉, 강화된 detector를 활용한 추가 학습이 모든 평가에서 일관된 성능 향상을 만들지는 않았다.

G2의 전체 의미 유사도·PPL 품질 평가는 수행하지 않았다.
사람 평가에서 Human/G0/G1/G2의 사람 글 판단율은 **70/68/64/58%**였으며, 추가 학습에 따른 증가는 확인하지 못했다.
[사람 평가 결과](../results/reports/human_eval_results_share_20260907.md)

## 실행

실행 전 위의 뉴스 19,738쌍과 G1·D0 checkpoint를 준비한다.
단계별 명령은 [adversarial/README.md](../adversarial/README.md), 상세 결과와 산출물 해시는 [RESULTS.md](../adversarial/RESULTS.md)에 있다.
