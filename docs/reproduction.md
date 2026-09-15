# 실행 가이드

모든 명령은 저장소 루트에서 실행합니다. 원천 데이터와 학습된 detector·generator
checkpoint는 별도로 준비합니다. 데이터 출처와 형식은 [데이터 안내](data.md)를 참고하세요.

## 환경 설정

Python 3.10 이상과 NVIDIA GPU를 사용합니다. CUDA/드라이버에 맞는 PyTorch를 설치한 뒤
주 파이프라인의 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

주 파이프라인은 Transformers 4.x를 사용합니다. `kiwipiepy`는 문장 분할·반복 진단,
`matplotlib`과 `scikit-learn`은 그래프/t-SNE, `bert-score`는 선택 분석 도구에 사용합니다.

Adversarial alignment는 Transformers 5.15.0을 사용하는 **별도 가상환경**에서 실행합니다.
설치 방법은 [`adversarial/README.md`](../adversarial/README.md)를 참고하세요.

Dockerfile은 PyTorch 2.5.1/cu121과 주 파이프라인 의존성을 설치합니다. 코드는 실행 시
마운트하며, llama-server·GGUF 모델·데이터·checkpoint는 별도로 연결합니다.

```bash
docker build -t aiku-humanization .
docker run --rm -it --gpus all -v "$PWD:/workspace" aiku-humanization
```

## 뉴스 데이터 준비

국립국어원 신문 말뭉치 2022판의 JSON ZIP에서 인간 원문을 추출합니다.
`--n`은 추출할 기사 수입니다.

```bash
python scripts/mash_extract_pool.py \
  --corpus /path/to/NIKLNEWSPAPER_2022_v1.0_JSON.zip \
  --n 24500 \
  --out /path/to/workspace/dataset/news_track/human_pool.jsonl

cp pipeline/configs/news.example.json pipeline/configs/news.local.json
```

다른 도메인도 `doc_id`와 본문을 포함한 `human_pool.jsonl`을 준비해 같은 파이프라인을
사용합니다. 그룹 분할과 대화 처리 설정은 [방법론](methodology.md), 입력 형식은
[`pipeline/README.md`](../pipeline/README.md)에 설명되어 있습니다.

## 주 파이프라인 실행

Config에 데이터·산출물 경로와 Qwen3-8B·EXAONE-3.5-7.8B endpoint를 설정합니다.
각 모델은 OpenAI-compatible llama-server로 먼저 실행합니다. `generate.py`의 `--model`은
결과에 기록할 모델 이름이며, 실제 모델은 endpoint에서 선택합니다. Qwen3은
`no_think=true`를 사용합니다.

```bash
python pipeline/run.py \
  --config pipeline/configs/news.local.json \
  --workspace /path/to/workspace \
  --gpu 0 --dry-run

python pipeline/run.py \
  --config pipeline/configs/news.local.json \
  --workspace /path/to/workspace \
  --gpu 0
```

`--dry-run`은 실행할 명령만 출력합니다. 단계는
`prompts → generate → gate → sft → hard-negative → dpo → evaluate` 순서입니다.
상대 산출물 경로는 workspace 기준이며, 조건별로 출력 경로를 나누어 실행합니다.

### 여러 게이트 결과 병합

주 뉴스 실험은 여러 게이트 결과를 합친 뒤 detector 학습 문서 제외, 기사 중복 제거,
분할을 거친 20,343쌍을 사용했습니다. Train 16,266 / dev 2,033 / test 2,044쌍입니다.

`gate`는 생성기별 통과 pair를 출력합니다. 기사당 한 pair로 병합하려면 아래 단계를
별도로 실행하고, 결과 경로를 config의 `paths.pairs`에 지정합니다. `--pairs`에는
먼저 유지할 파일부터 순서대로 지정합니다.

```bash
python scripts/build_clean_dataset.py \
  --pairs /path/to/core.jsonl /path/to/extension.jsonl /path/to/topup.jsonl \
  --exclude-detector /path/to/detector_split.json \
  --split 8:1:1 --seed 42 \
  --out /path/to/workspace/dataset/news_track/news_dpair_clean20k.jsonl
```

### 학습·평가 단계부터 실행

최종 pair와 동결 detector를 준비한 경우 `paths.pairs`, `paths.detector`를 지정하고
`--from-stage sft`로 시작합니다. 학습된 checkpoint의 평가만 실행하려면
`--from-stage evaluate`를 사용합니다.

| Config 경로 | 필요한 파일 |
| --- | --- |
| `paths.pairs` | `split`, `id`, `human_text`, `ai_text`를 포함한 최종 pair JSONL |
| `paths.detector` | 동결 RoBERTa 모델과 tokenizer 디렉터리 |
| `paths.sft` | `style_bart.pt`가 있는 디렉터리 |
| `paths.dpo` | `dpo_bart.pt`가 있는 디렉터리 |

## SCRN 평가

SCRN은 KoELECTRA 기반의 별도 평가용 detector입니다. 학습에는 `human_text`, `ai_text`,
`split` 필드를 포함한 SCRN용 train/dev pair를 지정합니다.

```bash
python scripts/scrn_train.py \
  --pairs /path/to/scrn_train_dev_pairs.jsonl \
  --out-dir /path/to/workspace/models/news_scrn \
  --epochs 3 --bs 8 --accum 2
```

평가 config의 `evaluation.scrn`에 `scrn_best.pt` 경로를 지정하면 RoBERTa와 함께 채점합니다.
기본값은 `null`입니다. 학습 스크립트는 `--pairs`에 지정한 train/dev split을 그대로 사용하므로
SCRN용 분할을 입력 파일에 구성합니다.

## 교차 도메인 평가

**Source의 generator와 target의 pair·detector·SCRN**을 조합합니다.
다음은 뉴스 generator를 에세이 test에서 평가하는 명령입니다.

```bash
python pipeline/evaluate.py \
  --domain essay \
  --pairs /path/to/essay/dpair.jsonl \
  --detector /path/to/essay/roberta \
  --scrn /path/to/essay/scrn_best.pt \
  --sft /path/to/news/style_bart.pt \
  --dpo /path/to/news/dpo_bart.pt \
  --n 0 \
  --out /path/to/workspace/logs/transfer/news_to_essay.jsonl
```

Source/target 경로와 출력 파일을 바꾸어 6×6 조합을 실행합니다. Persona target에는
`--dialogue`를 추가합니다. ASR 임계값은 target Human 점수의 P95이며, 같은 target에서
SFT와 DPOP를 비교합니다. 교차 도메인 결과는 SFT와 첫 DPOP generator를 대상으로 합니다.

## Prompting baseline과 추가 실험

Prompting baseline은 SFT/DPOP와 같은 문서 ID를 사용합니다. 고정 프롬프트, reference
파일과 20→100편 실행 절차는 [`baselines/README.md`](../baselines/README.md)에 있습니다.

강화된 detector D1과 추가 DPOP generator G2의 학습은
[`adversarial/README.md`](../adversarial/README.md)를 따릅니다. 이 실험은 주 실험의
2,044편 test와 별도로 구성한 1,985편 뉴스 test에서 G1/G2를 비교합니다.

결과 도표는 [`scripts/figures/`](../scripts/figures/)의 스크립트로 재생성합니다.
