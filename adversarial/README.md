# Adversarial Alignment — 뉴스 G1 → D1 → G2

**Generator의 출력에 적응한 detector를 학습 신호로 사용하면, generator의 회피 성능을
더 높일 수 있을까?** 이를 확인하기 위해 기존 뉴스 generator G1의 출력으로 detector
D1을 학습하고, D1 점수를 활용한 추가 DPOP로 G2를 학습했다.

실험은 **G1 → D1 → G2 한 사이클**로 진행했다. 방법은
[adversarial alignment](../docs/adversarial_alignment.md), 결과는 [RESULTS.md](RESULTS.md)에 정리했다.

## 실험 설정

| 이름 | 역할 |
| --- | --- |
| G0 | 원래 AI 생성문. generator의 입력 |
| G1 | SFT와 첫 DPOP를 완료한 KoBART generator |
| D0 | 기존 뉴스 RoBERTa detector. 평가 시 고정 |
| D1 | Human/G0/G1 train 글로 학습한 RoBERTa detector |
| G2 | D1 점수를 활용해 G1을 DPOP로 추가 학습한 generator |

초기 checkpoint는 `news_dpo_D2`와 `news_roberta_D2`이며 각각 G1, D0에 해당한다.
파일명에 붙은 `D2`는 이 실험의 반복 인덱스와 무관하다.

총 **19,738쌍(train 15,780 / dev 1,973 / test 1,985)**의 뉴스 데이터를 사용했다.
이후 비교는 동일한 test ID 1,985개에서 수행했다.

| 조건 | 주 실험 | Adversarial alignment |
| --- | --- | --- |
| 전체 pair 수 | 20,343 | 19,738 |
| Test pair 수 | 2,044 | 1,985 |
| Decoding | Beam 4 | Greedy |
| 입력 최대 길이 | 512토큰 | 512토큰 |
| 출력 최대 길이 | 1,024토큰 | 512토큰 |

데이터 버전과 decoding 조건이 달라 두 실험의 ASR을 직접 비교하지 않는다.

### 학습 방법

1. **D1 학습:** G1으로 train의 G0를 재작성한다. Human/G0/G1에 총 가중치
   **50%/25%/25%**를 부여해 사전학습 RoBERTa에서 D1을 학습한다.
2. **Threshold 설정:** dev를 문서 ID 기반으로 dev-cal/dev-gate로 나눈다.
   cal에서 epoch와 Human FPR 5% threshold를 정하고, gate에서 Human FPR ≤ 8%,
   G0 TPR ≥ 80%를 확인한다.
3. **Preference 구성:** G1 후보를 문서당 8개, 필요하면 16개까지 생성한다.
   D1 점수가 threshold보다 높은 후보를 최대 2개 rejected로 선택하고, 인간 원문을
   chosen으로 사용한다.
4. **G2 학습:** 입력은 G0, reference는 G1이다. Reference log-prob를 미리 계산한 뒤,
   G1에서 시작한 policy를 token 평균 log-prob 기반 DPOP로 학습한다.

DPOP 설정은 **β=2, λ=5, LR=5e-6, 1 epoch, 실효 batch 16**이다.
`configs/base.yaml`에 `configs/arms/team_news.yaml`을 적용해 실행한다.

```text
h = (log G2(chosen) - log G1(chosen))
    - (log G2(rejected) - log G1(rejected))
    - lambda * max(0, log G1(chosen) - log G2(chosen))
loss = -log sigmoid(beta * h)
```

## 뉴스 실험 재현

### 데이터와 환경 준비

아래 데이터와 checkpoint를 별도로 준비한다. 데이터 출처와 취득 방법은
[데이터 안내](../docs/data.md)를 참고한다.

| `--team-root` 기준 경로 | 내용 |
| --- | --- |
| `dataset/news_track/news_dpair_D2.jsonl` | `id`, `split`, `human_text`, `ai_text`가 있는 19,738쌍 |
| `models/news_dpo_D2/dpo_bart.pt` | G1 checkpoint |
| `models/news_roberta_D2/` | D0의 Hugging Face 모델·tokenizer 파일 |

실행 환경은 **Linux CUDA**다. 주 파이프라인의 Transformers 4.x 환경과 분리해
가상환경을 만들고, GPU에 맞는 PyTorch를 설치한다. 아래 명령은 저장소 루트에서 실행한다.

```bash
python3 -m venv .venv-adversarial
source .venv-adversarial/bin/activate
# GPU에 맞는 PyTorch 설치 후 실행
pip install -r adversarial/requirements.txt
python adversarial/check_env.py --profile team-news --team-root .

# Pair 데이터와 초기 checkpoint를 불러온다.
python adversarial/scripts/import_team_news.py --team-root .

# G1 기준선 평가
python adversarial/scripts/r_eval.py \
  --arm-config adversarial/configs/arms/team_news.yaml \
  --round 0 --device cuda --light --no-ck-export

# D1 학습
python adversarial/scripts/bootstrap_retrain_detector.py --device cuda

# 8문서로 실행을 점검한 뒤 G2 전체 학습·평가
python -u adversarial/scripts/run_team_news_g2.py --smoke
python -u adversarial/scripts/run_team_news_g2.py
```

`import_team_news.py`의 D0 threshold 기본값은 **0.062**로, 기존 D0에서 Human FPR
5%를 기준으로 정한 값이다. 다른 D0를 사용할 때는 해당 모델의 calibration 결과를
`--tau`로 전달한다. `run_team_news_g2.py`는 **train 15,780 / test 1,985**를 전제로 한다.

`--smoke`는 실제 G1/D1을 사용해 train 문서 8개로 실행을 점검한다.
전체 실행 결과는 `adversarial/runs/team_news/arms/team_news/round1/`에 생성된다.
G2 checkpoint는 `dpo/final/`, 평가 결과는 `eval/metrics.json`에서 확인할 수 있다.

### 평가

`metrics.json`의 `frozen_d0`는 D0 평가, `in_loop_dt`는 D1 평가다.
D1은 G2 학습에 사용한 detector이므로, 독립 detector로의 전이는 CopyKiller에서 평가했다.
`r_eval.py`는 round 0에서 G1-D0, round 1에서 G2-D0/D1 평가를 지원하며,
G1-D1 비교는 별도 평가가 필요하다.

G2 실행은 `--light`를 사용해 **SimCSE/PPL 평가를 생략**한다.
G2 출력을 학습한 **새 D2 평가는 수행하지 않았다**.

Human/G0/G1/G2 텍스트 예시는 다음 명령으로 추출한다.

```bash
python adversarial/scripts/export_team_news_review.py --limit 20
```

### CopyKiller 평가

동일한 test ID 1,985개에 대해 Human/G0/G1/G2를 각각 평가한다.
G1/G2에서 선택한 ID를 Human/G0에도 그대로 적용한다.

```bash
python adversarial/scripts/export_team_news_copykiller.py \
  --g1 adversarial/runs/team_news/arms/team_news/round0/eval/gen_test.jsonl.gz \
  --g2 adversarial/runs/team_news/arms/team_news/round1/eval/gen_test.jsonl.gz \
  --out adversarial/runs/team_news/copykiller/export_full_g1_g2 \
  --limit 1985 --batch-size 350 --seed 42

python adversarial/scripts/export_team_news_copykiller_baselines.py \
  --pairs data/team_news/news_dpair_D2.jsonl.gz \
  --selection-metadata adversarial/runs/team_news/copykiller/export_full_g1_g2/export_metadata.json \
  --out adversarial/runs/team_news/copykiller/export_full_human_g0
```

각 변형은 서로 다른 검사로 제출한다. 같은 기사의 변형들이 서로 표절 대조되는 것을
방지하기 위해서다. 제출 대상은 생성된 DOCX이며, `manifest.csv`와
`export_metadata.json`은 분석용 대응표다.
PDF 결과 분석 방법은 [RESULTS.md](RESULTS.md#결과-집계)에 있다.

## 코드 구성

| 위치 | 역할 |
| --- | --- |
| `configs/` | 뉴스 실험과 추가 ablation 설정 |
| `loop_lib/` | 데이터, detector, generator, SFT, DPOP, 평가 모듈 |
| `scripts/import_team_news.py`, `bootstrap_retrain_detector.py`, `run_team_news_g2.py` | G1 → D1 → G2 실행 |
| `scripts/r_*.py` | 후보 생성, preference 구성, 학습과 평가 |
| `scripts/export_*`, `validate_*`, `analyze_*` | 텍스트 추출, CopyKiller 입력 검증과 통계 분석 |

## 추가 실험 설정

초기 KCI 초록 실험과 반복 ablation을 위한 코드도 포함한다.
위 뉴스 결과에는 이 ablation들의 비교 결과를 포함하지 않았다.

| 코드·설정 | 실험 내용 |
| --- | --- |
| `s0_train_detector.py` → `s1_build_dpair.py` → `s2_sft.py` | KCI detector 학습, OOF 점수 기반 pair 선별, SFT |
| `run_round.sh`, `run_arm.sh` | Generator와 detector를 여러 라운드 반복 학습 |
| `main`, `self_anchor` | 인간 원문과 자기 생성문 chosen 비교 |
| `continual` | Detector를 재초기화하지 않고 연속 학습 |
| `latest_only` | 최신 generator 출력만 replay에 사용 |
| `build_p4_prompts.py`, `s0_probe_external.py` | KCI 문체 prompting과 OOD 평가 |

KCI 경로는 `configs/base.yaml`의 데이터 파일을 준비한 뒤
`bash adversarial/scripts/run_stage0_2.sh`와
`bash adversarial/scripts/run_arm.sh main 1 5`로 실행한다.
Detector gate 기준을 벗어나면 해당 arm의 학습을 중단한다.
