# 팀 뉴스 G1 → D1 → G2 결과

기준일은 2026-09-05다. 원문, 생성문, 체크포인트, CopyKiller PDF는 저작권과
서비스 약관 때문에 Git에 포함하지 않는다. 이 문서는 로컬에 보존된 원시 산출물을
공개 가능한 집계와 SHA256으로 요약한다.

## 실험 계보

| 단계 | 학습 또는 처리 | 출력 |
| --- | --- | --- |
| import | 팀 `news_dpo_D2`와 `news_roberta_D2`를 루프 형식으로 변환·복사 | G1, frozen D0 |
| detector adaptation | human/G0/G1 train 출력을 50%/25%/25% 가중으로 학습 | D1 |
| generator adaptation | D1이 탐지한 hard negative, human chosen, G1 reference로 DPOP | G2 |
| evaluation | 동일 test ID 1,985개를 frozen/in-loop/external detector로 paired 비교 | 아래 표 |

팀 모델명의 `D2`와 반복 인덱스를 혼동하면 안 된다. 현재 코드는 G2 평가에서 멈추며
새 반복 탐지기 D2를 학습하지 않았다.

## D1 게이트

D1은 train 15,780개에서 학습했다. threshold는 dev-cal에서 정하고, 겹치지 않는
dev-gate 951개에서 붕괴 여부를 판정했다.

| 항목 | 결과 | 사전 기준 |
| --- | ---: | ---: |
| 선택 epoch의 dev-cal AUC | 0.999686 | 최대 AUC epoch 선택 |
| threshold | 0.9996544122695923 | dev-cal human FPR 목표 5% |
| dev-gate human FPR | 6.31% | ≤ 8% |
| dev-gate original-AI TPR | 100.00% | ≥ 80% |
| collapsed | false | false |

같은 test 1,985개에서 G1 ASR은 frozen D0 기준 93.90%, D1 기준 3.38%였다. 이는
D1이 G1 train 출력을 학습한 뒤의 **in-domain adaptation** 결과이지 독립 일반화
증거가 아니다.

## G1과 G2 비교

아래 G2의 frozen team detector와 D1 수치는 2026-08-29 서버에서
`g2_status.json`, checkpoint, `eval/metrics.json`을 함께 확인해 기록한 값이다.
원시 서버 산출물은 공개 저장소에 포함되지 않으므로, 팀원이 Drive 묶음을 복원한 뒤
위 세 완료 증거와 해시를 다시 확인해야 한다. 아래 CopyKiller 집계는 현재 로컬에
보존된 PDF와 manifest로 2026-09-05 재계산했다.

| 평가 축 | G1 ASR | G2 ASR | G2−G1 | 해석 |
| --- | ---: | ---: | ---: | --- |
| frozen team detector | 93.90% | 92.64% | −1.26%p | 고정 앵커에서는 개선 없음 |
| D1 | 3.38% | 3.93% | +0.55%p | 학습 신호원과 같은 탐지기라 순환 지표 |
| 새 D2 | — | — | — | 아직 학습·평가하지 않음 |

ASR은 탐지 회피율이며 자연스러움, 사실성, 의미 보존을 측정하지 않는다. D1과 기존
team detector 결과만으로 G2의 전체 인간화 품질을 판정하지 않는다.

## CopyKiller 전수 paired 평가

동일 test ID 1,985개를 Human/G0/G1/G2 각각 별도 검사로 제출하고 AI작성률 50%를
AI 판정 threshold로 사용했다.

| 변형 | 평균 AI작성률 | AI 판정률 | ASR |
| --- | ---: | ---: | ---: |
| Human | 1.303% | 0.86% (17/1,985) | — |
| G0 | 39.258% | 40.20% (798/1,985) | 59.80% |
| G1 | 3.913% | 2.97% (59/1,985) | 97.03% |
| G2 | 3.285% | 2.17% (43/1,985) | 97.83% |

G1→G2 평균 AI작성률 변화는 −0.628%p(bootstrap 95% CI −1.084~−0.176%p),
ASR 변화는 +0.806%p였다. threshold 전이의 exact McNemar p=0.01133이지만,
점수가 바뀐 문서의 방향성 sign-test p=0.16346이다. 따라서 G2의 추가 효과는
**작고 지표 선택에 민감하다**고 해석한다.

파일럿과 전수 검사에 겹친 content-identical DOCX 700개 중 692개 점수만 정확히
일치했고 8개는 최대 46%p 차이가 났다. CopyKiller 점수를 완전히 결정적인 값으로
취급하지 않고, 검사 PDF와 배치 구성을 함께 보존한다.

## 무결성과 재계산

| 비공개 증거 | SHA256 |
| --- | --- |
| Human 결과 PDF | `202e6c532d2fda625846207dc323db05952e279cf3364b67160f3076dc50f3de` |
| G0 결과 PDF | `32288df3944bd484bdd2ecc6646b24f47086ec5417d8a3a39d70e5eda87aeead` |
| G1 결과 PDF | `0e5b6ecd84e260a76a1a384ef6c05cd74d44a48ad326879512e282940692747e` |
| G2 결과 PDF | `f21ae2b48e7c7de9ed7dff725a0abacfed6a55c919e2fbe87c0e2cc87a0acd31` |
| G1 test 생성 JSONL.GZ | `8a310c0038ddb1c4ffc0f7e37e081d1b0c5baf2e1460ecf24c7dacf05dce87b5` |
| G2 test 생성 JSONL.GZ | `ee216b8e31c243d6a8ffa8d89af8b7807e5e53d3455bbf6ee77181ed73431cae` |

Drive 산출물을 `loop/runs/team_news/copykiller/` 아래에 복원한 뒤 다음 순서로
재계산한다. G1/G2와 Human/G0 export는 서로 다른 검사로 올리고, 어떤 경우에도
두 변형의 DOCX를 한 검사에 섞지 않는다.

```bash
python loop/scripts/validate_team_news_copykiller_export.py \
  --root loop/runs/team_news/copykiller \
  --export loop/runs/team_news/copykiller/export_full_g1_g2

python loop/scripts/validate_team_news_copykiller_baselines.py \
  --pairs data/team_news/news_dpair_D2.jsonl.gz \
  --selection-metadata loop/runs/team_news/copykiller/export_full_g1_g2/export_metadata.json \
  --export loop/runs/team_news/copykiller/export_full_human_g0

python loop/scripts/analyze_team_news_copykiller_full.py \
  --human-pdf '<Human 결과확인서.pdf>' \
  --g0-pdf '<G0 결과확인서.pdf>' \
  --g1-pdf '<G1 결과확인서.pdf>' \
  --g2-pdf '<G2 결과확인서.pdf>' \
  --baseline-manifest loop/runs/team_news/copykiller/export_full_human_g0/manifest.csv \
  --generator-manifest loop/runs/team_news/copykiller/export_full_g1_g2/manifest.csv \
  --selection-metadata loop/runs/team_news/copykiller/export_full_g1_g2/export_metadata.json \
  --out-dir loop/runs/team_news/copykiller/results \
  --tau 50 --bootstrap-samples 50000 --seed 42
```

## 아직 필요한 검증

- G2 train 출력으로 새 D2를 학습하고 dev-cal/dev-gate를 분리한 paired G1/G2 평가
- 동일 test ID에서 SimCSE와 PPL을 포함한 full quality evaluation
- 수치·방향·고유명사 보존과 사실성 검수
- variant명과 detector score를 가린 사람 자연스러움 평가

이 네 항목 전에는 “G2가 전반적으로 더 자연스럽다”거나 “반복 학습이 성공했다”고
결론내리지 않는다.
