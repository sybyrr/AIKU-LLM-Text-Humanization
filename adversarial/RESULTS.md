# Adversarial Alignment 결과

**G1 출력에 적응한 D1을 학습 신호로 사용했을 때, G2의 추가 회피 효과는 제한적이었다.**
D1과 CopyKiller에서는 ASR이 소폭 증가했지만, 고정된 D0에서는 감소했다.

## 실험 조건

| 단계 | 방법 |
| --- | --- |
| 초기 모델 | 뉴스 G1(`news_dpo_D2`)과 D0(`news_roberta_D2`) |
| D1 학습 | Human/G0/G1 train 출력에 총 가중치 50%/25%/25% 적용 |
| G2 학습 | D1이 탐지한 후보를 rejected, 인간 원문을 chosen, G1을 reference로 DPOP |
| 비교 | 동일 test ID 1,985개에서 G1과 G2 평가 |

총 **19,738쌍(train 15,780 / dev 1,973 / test 1,985)**을 사용했고, 평가 시
**greedy decoding, 입력/출력 최대 512토큰**을 적용했다.
주 실험은 test 2,044편, beam 4, 출력 최대 1,024토큰으로 조건이 달라 직접 비교하지 않는다.

초기 모델 파일명의 `D2`는 반복 인덱스와 무관하다. 이 실험은 G2 평가까지 진행했으며,
G2 출력으로 학습한 새 D2는 평가하지 않았다.
실행 방법은 [README.md](README.md#뉴스-실험-재현)에 있다.

## D1 학습 결과

D1은 train 15,780편의 Human/G0/G1으로 학습했다. Dev 1,973편을
**cal 1,022편 / gate 951편**으로 나누어, cal에서 epoch와 threshold를 선택하고
별도 gate에서 Human FPR과 원본 AI 탐지율을 측정했다.

| 항목 | 결과 | 기준 |
| --- | ---: | --- |
| 선택 epoch의 dev-cal AUC | 0.999686 | 최대 AUC epoch 선택 |
| Threshold | 0.9996544122695923 | dev-cal Human FPR 목표 5% |
| dev-gate Human FPR | 6.31% | ≤ 8% |
| dev-gate G0 TPR | 100.00% | ≥ 80% |

동일 test 1,985편에서 G1의 ASR은 **D0 기준 93.90%, D1 기준 3.38%**였다.
D1은 G1 train 출력을 학습한 뒤, 보지 않은 G1 test 출력을 높은 비율로 탐지했다.

## G1과 G2 비교

D0와 D1의 ASR은 generator 출력 점수가 각 detector의 threshold 이하인 비율이다.

| Detector | G1 ASR | G2 ASR | G2−G1 |
| --- | ---: | ---: | ---: |
| D0 | 93.90% | 92.64% | −1.26%p |
| D1 | 3.38% | 3.93% | +0.55%p |

D1을 활용한 추가 DPOP 이후 D1 ASR은 **0.55%p 증가**했지만, D0 ASR은
**1.26%p 감소**했다. 강화된 detector의 점수로 generator를 다시 학습하는 것만으로
일관된 회피 성능 향상을 얻지는 못했다.

D1은 G2 학습에 사용한 detector다. 독립 detector에 대한 전이 효과는 아래
CopyKiller 실험에서 평가했다. `r_eval.py`의 평가 대상은 round 0에서 G1-D0,
round 1에서 G2-D0/D1이며, G1-D1 비교는 별도 평가가 필요하다.

## CopyKiller 결과

동일 test ID 1,985개를 Human/G0/G1/G2 각각 별도 검사로 제출했다.
**AI작성률 ≥ 50%**를 AI 판정으로, **50% 미만**을 회피 성공으로 계산했다.

| 변형 | 평균 AI작성률 | AI 판정률 | ASR |
| --- | ---: | ---: | ---: |
| Human | 1.303% | 0.86% (17/1,985) | — |
| G0 | 39.258% | 40.20% (798/1,985) | 59.80% |
| G1 | 3.913% | 2.97% (59/1,985) | 97.03% |
| G2 | 3.285% | 2.17% (43/1,985) | 97.83% |

G1→G2 평균 AI작성률 변화는 **−0.628%p**(bootstrap 95% CI −1.084~−0.176%p),
ASR 변화는 **+0.806%p**였다. 임계값 판정 변화의 exact McNemar p=0.01133인 반면,
점수가 바뀐 문서의 방향성 sign-test p=0.16346이었다.
추가 학습의 효과는 작았고, 지표에 따라 통계적 근거의 강도가 달랐다.

### 재검사 시 점수 변동

파일럿과 전수 검사에 공통으로 사용한 동일 내용의 DOCX 700개 중 **692개 점수가
일치**했고, 8개는 **최대 46%p 차이**가 났다. 같은 입력에서도 재검사에 따른 점수
변동이 관찰되어, CopyKiller의 작은 점수 차이를 해석할 때 이를 고려했다.

## 한계

- **반복 효과:** G1 → D1 → G2 한 사이클만 실험했다. G2에 적응한 새 D2에서의 성능과
  여러 사이클을 거친 변화는 평가하지 않았다.
- **글의 품질:** G2 평가는 `--light`로 실행해 SimCSE/PPL을 생략했다. ASR은 회피
  성능을 측정하며, 자연스러움·사실성·의미 보존의 향상을 직접 보여주지는 않는다.
- **외부 전이:** CopyKiller 한 서비스에서 평가했고, 재검사 시 점수 변동도 관찰됐다.

## 결과 집계

CopyKiller에서 받은 네 결과 PDF를 export 시 생성한 manifest에 조인한다.
Human/G0/G1/G2 모두 동일한 1,985개 ID가 포함되어야 한다.

입력 DOCX 검증에는 export에 사용한 G1/G2 생성 파일을 다음 경로에 준비한다.

```text
adversarial/runs/team_news/copykiller/source/g1_gen_test.jsonl.gz
adversarial/runs/team_news/copykiller/source/g2_gen_test.jsonl.gz
```

```bash
python adversarial/scripts/validate_team_news_copykiller_export.py \
  --root adversarial/runs/team_news/copykiller \
  --export adversarial/runs/team_news/copykiller/export_full_g1_g2

python adversarial/scripts/validate_team_news_copykiller_baselines.py \
  --pairs data/team_news/news_dpair_D2.jsonl.gz \
  --selection-metadata adversarial/runs/team_news/copykiller/export_full_g1_g2/export_metadata.json \
  --export adversarial/runs/team_news/copykiller/export_full_human_g0

python adversarial/scripts/analyze_team_news_copykiller_full.py \
  --human-pdf '<Human 결과확인서.pdf>' \
  --g0-pdf '<G0 결과확인서.pdf>' \
  --g1-pdf '<G1 결과확인서.pdf>' \
  --g2-pdf '<G2 결과확인서.pdf>' \
  --baseline-manifest adversarial/runs/team_news/copykiller/export_full_human_g0/manifest.csv \
  --generator-manifest adversarial/runs/team_news/copykiller/export_full_g1_g2/manifest.csv \
  --selection-metadata adversarial/runs/team_news/copykiller/export_full_g1_g2/export_metadata.json \
  --out-dir adversarial/runs/team_news/copykiller/results \
  --tau 50 --bootstrap-samples 50000 --seed 42
```

문서별 점수는 `copykiller_scores_full.jsonl`, paired 점수는 `paired_scores_full.csv`,
집계 결과는 `analysis.json`과 `report.md`로 출력된다.
