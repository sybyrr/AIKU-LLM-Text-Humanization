# AIKU Korean LLM Text Humanization

한국어 AI 생성문을 사람 문체로 재서술하는 StyleBART 기반 연구 파이프라인입니다.
도메인별 탐지기로 학습 pair를 구성하고 SFT와 DPOP를 거친 뒤 held-out test에서
탐지 회피와 반복 붕괴를 평가합니다.

> 이 저장소에는 코드와 문서만 포함합니다. 인간 원문, 생성문, checkpoint, 평가 로그와
> 검수 HTML은 저작권·데이터 이용약관 때문에 배포하지 않습니다.

## Canonical pipeline

```text
human_pool.jsonl
  → P3b 공통 프롬프트
  → Qwen·EXAONE 재서술
  → frozen domain detector gate
  → StyleBART SFT
  → hard-negative mining
  → DPOP
  → held-out test plain-beam evaluation
```

현재 정본은 [pipeline/README.md](pipeline/README.md)에 정의되어 있습니다.
`loop/`는 generator와 detector를 반복 재학습하는 별도 실험이며 위 정본 파이프라인과
설정이나 결과를 합치지 않습니다.

## Repository layout

```text
pipeline/                 정본 설정, 오케스트레이터, 최종평가, 반복 지표
pipeline/configs/         공개 가능한 도메인 설정 예시
baselines/                프롬프트 기준선 명세, 고정 프롬프트, 설정 예시
scripts/                  정본 단계 구현과 과거 연구용 도구
loop/                     별도 generator↔detector 반복학습 실험
notes/                    실험 설계·결과·시행착오 기록
mash/                     데이터셋 및 프롬프트 명세
```

정본 단계 구현은 다음 파일입니다.

- `scripts/build_domain_prompts.py`
- `scripts/generate.py`
- `scripts/domain_gate.py`
- `scripts/stage2_sft.py`
- `scripts/stage3_build_dpo.py`
- `scripts/stage3_dpo.py`
- `pipeline/evaluate.py`

그 밖의 t-SNE, cross-domain, LLM judge, HTML 검수 코드는 분석·실험 도구이며 정본
학습 경로에 자동으로 들어가지 않습니다.

## Environment

Python 3.10 이상과 CUDA 환경이 필요합니다. GPU에 맞는 PyTorch를 먼저 설치한 뒤:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

생성 단계는 OpenAI-compatible `/v1/chat/completions` endpoint를 사용합니다.
Qwen과 EXAONE 서버 주소는 설정 파일에 각각 지정합니다.

## Input

도메인별 `human_pool.jsonl`의 최소 스키마는 다음과 같습니다.

```json
{"doc_id":"unique-id","text":"사람이 작성한 원문","n_char":934}
```

본문은 `text`, `body`, `human_text` 중 하나를 사용할 수 있습니다. 동일 논제·도서·대화가
train/test에 함께 들어가면 안 되는 도메인은 그룹 필드를 추가하고 `gate.split_key`로
지정합니다. 대화 데이터는 `n_turn`과 `gate.dialogue=true`를 사용합니다.

## Run

신문 플래그십 설정을 복사해 비공개 경로와 생성 서버 주소를 수정합니다.

```bash
cp pipeline/configs/news.example.json pipeline/config.news.json

python pipeline/run.py \
  --config pipeline/config.news.json \
  --gpu 0 \
  --dry-run

python pipeline/run.py \
  --config pipeline/config.news.json \
  --gpu 0
```

개별 단계만 실행할 수 있습니다.

```bash
python pipeline/run.py \
  --config pipeline/config.news.json \
  --gpu 0 \
  --from-stage sft \
  --to-stage evaluate
```

기본 실행은 완료 산출물을 검증하고 이어서 실행합니다. 로그와 상태 파일은
`logs/{domain}/pipeline/`에 기록됩니다. 로컬 설정, 데이터, 모델과 로그는 `.gitignore`
대상입니다.

## Canonical settings

| 단계 | 설정 |
| --- | --- |
| SFT | ko-BART, concat→projection fusion, target EOS, max length 512, λ=0.5, 3 epochs |
| Hard negative | SFT sampling, D score > τ, 4 candidates, 기존 재현값 `no_repeat_ngram_size=3` |
| DPO | length normalization, β=2, DPOP λ=5, 1 epoch, effective batch 16 |
| Final generation | plain beam4, input max 512, output max 1024, 반복방지 옵션 없음 |

`hard_negative.no_repeat_ngram_size=0`은 붕괴 출력도 rejected 후보로 노출하는 별도
실험 조건입니다. 기존 결과를 재현할 때는 기본값 3을 유지합니다.

## Evaluation

최종평가는 pair 파일의 test split만 사용합니다.

- 주 지표: 도메인 인간 원문 P(AI)의 95분위 임계에서 raw ASR
- 안전성: 공백 제거 문자 6-gram이 20회를 초과해 반복되는 급성붕괴율
- 의미 보존: ko-sroberta cosine
- 선택 평가: 동일 detector split으로 학습한 SCRN

collapse는 raw ASR에서 차감하지 않고 별도로 보고합니다. LLM-as-judge 품질 점수는
모델별 편차가 커 정본 평가에서 제외합니다. 평가기는 문서별 JSONL과 집계
`*.summary.json`을 함께 저장합니다.

## Prompt baseline and zero-shot detectors

학습을 사용하지 않는 비교군으로 Humanizer-skill 프롬프팅 기준선을 평가합니다. 같은
held-out `x_ai`를 기존 SFT, 기존 DPO, Humanizer-skill 프롬프트에 각각 독립적으로
입력합니다. 프롬프트 출력에 SFT나 DPO를 다시 적용하지 않습니다.

Binoculars와 FastDetectGPT는 도메인 탐지기 학습에 사용하지 않은 zero-shot 평가기입니다.
6×6 크로스도메인 전량과 Humanizer-skill 표본에 같은 target별 FPR 5% 기준을 적용합니다.
분산 채점 결과는 한 행씩 저장되며, 중단 시 완료된 행 다음부터 재개할 수 있습니다.

실험 조건, 파일럿 게이트, 실행 방법과 외부 프롬프트 출처는
[baselines/README.md](baselines/README.md)에 정리되어 있습니다.

## CPU checks

GPU를 쓰지 않고 설정과 command graph, 재개 판정, 표본 추출 및 반복 지표를 검사합니다.

```bash
python -m compileall -q pipeline scripts
python -m unittest discover -s pipeline/tests -v
python pipeline/run.py --config pipeline/configs/news.example.json --dry-run
```

## Results

아래에는 공개 가능한 도메인의 **집계 결과만** 제시합니다. 데이터 이용 조건을 준수하기 위해
원문, 생성문, 문서 식별자, 문서별 점수와 검수 자료는 공개하지 않습니다.

- 공개 대상: `essay`, `persona`, `petition`, `wiki`
- 공개 보류: `news`, `written` — 국립국어원 말뭉치 결과물의 공개 절차를 확인한 뒤 공개 여부를 결정합니다.
- 비교 제외: KCI 논문 초록 Stage 0 — 아래의 SFT/DPO 실험과 평가 조건이 다릅니다.

### Data

`pair`는 도메인 탐지기 게이트를 통과한 인간 원문–AI 재서술 쌍의 수를 의미합니다.

| domain | source | pair | train | dev | test | evaluation input |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| essay | [AI Hub 에세이 글 평가 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&topMenu=100&dataSetSn=545) | 6,795 | 4,750 | 1,023 | 1,022 | Qwen |
| persona | [AI Hub 페르소나 대화](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&topMenu=100&dataSetSn=71302) | 11,169 | 8,935 | 1,117 | 1,117 | Qwen |
| petition | [Blue House National Petition](https://huggingface.co/datasets/dev7halo/bluehouse-national-petition) | 24,252 | 19,401 | 2,425 | 2,426 | Qwen + EXAONE |
| wiki | [Korean Wikipedia, 20231101.ko](https://huggingface.co/datasets/wikimedia/wikipedia) | 23,429 | 18,743 | 2,343 | 2,343 | Qwen + EXAONE |

essay와 persona 모델은 Qwen과 EXAONE으로 생성한 pair를 약 75:25 비율로 혼합하여 학습했습니다.
다만 모델 간 입력 조건을 동일하게 유지하기 위해 평가는 기존 Qwen test split 전체에서
수행했습니다. petition과 wiki는 확장된 혼합 pair의 held-out test split을 사용했습니다.

### Evaluation protocol

- 평가일: 2026-09-07
- 범위: 각 도메인의 held-out test split 전체
- 생성: beam search(`num_beams=4`), 입력 최대 512 token, 출력 최대 1,024 token, 반복 억제 옵션 미적용
- `D`: 데이터 게이트와 DPO 보상에 사용한 도메인별 KLUE-RoBERTa 탐지기
- `SCRN`: 동일한 detector split으로 별도 학습한 KoELECTRA 기반 검증 탐지기
- `ASR`: 해당 도메인의 인간 원문 점수 95분위를 임계값으로 적용했을 때 인간으로 판정된 비율
- `P(AI)`: 탐지기가 산출한 AI 확률의 평균입니다. 값이 낮을수록 인간 문서에 가깝게 판정됩니다.
- `cos`: ko-sroberta cosine(output, x_ai)입니다. 높은 값은 의미 보존뿐 아니라 입력 복사의 영향도 받을 수 있습니다.
- `acute`: 공백 제거 후 동일 문자 6-gram이 20회를 초과해 반복된 문서의 비율

### Detector calibration

ASR을 해석하기에 앞서 각 탐지기가 인간 원문과 AI 재서술문을 구분하는지 확인했습니다.

| domain | D human P(AI) | D x_ai P(AI) | SCRN human P(AI) | SCRN x_ai P(AI) |
| --- | ---: | ---: | ---: | ---: |
| essay | 0.006 | 0.996 | 0.049 | 0.986 |
| persona | 0.172 | 0.963 | 0.101 | 0.754 |
| petition | 0.009 | 0.979 | 0.019 | 0.976 |
| wiki | 0.031 | 0.993 | 0.036 | 0.983 |

모든 공개 도메인에서 두 탐지기 모두 human과 x_ai의 평균 `P(AI)` 격차가 0.4를 넘었습니다.

### SFT and DPO

| domain | model | D ASR | D P(AI) | SCRN ASR | SCRN P(AI) | cos | acute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| essay | SFT | 46.9% | 0.313 | 40.8% | 0.544 | 0.991 | 0.4% |
| essay | DPO | 47.2% | 0.323 | 39.2% | 0.558 | 0.991 | 0.9% |
| persona | SFT | 39.2% | 0.543 | 56.2% | 0.346 | 0.994 | 0.1% |
| persona | DPO | 53.9% | 0.450 | 67.6% | 0.269 | 0.993 | 0.5% |
| petition | SFT | 59.6% | 0.233 | 62.1% | 0.227 | 0.979 | 4.7% |
| petition | DPO | 62.9% | 0.213 | 65.3% | 0.198 | 0.976 | 5.5% |
| wiki | SFT | 52.8% | 0.390 | 65.2% | 0.285 | 0.978 | 7.3% |
| wiki | DPO | 68.6% | 0.252 | 77.9% | 0.175 | 0.977 | 9.4% |

`D`는 데이터 게이트와 DPO 최적화에 사용되었으므로 독립적인 일반화 성능을 나타내지 않습니다.
따라서 결과를 해석할 때는 별도로 학습한 `SCRN`의 측정값을 함께 고려해야 합니다. petition과
wiki의 DPO 모델은 ASR이 향상된 동시에 acute 비율도 각각 5.5%, 9.4%로 증가했습니다. 생성 품질은
탐지 회피율만으로 판단하지 않았으며, acute는 ASR에서 차감하지 않고 별도의 안전성 지표로
보고했습니다.

### Summary

- essay에서는 SFT와 DPO의 차이가 작았으며, `SCRN ASR`은 DPO에서 1.6%p 감소했습니다.
- persona에서는 DPO 적용 후 두 탐지기의 ASR이 모두 향상되었고 acute 비율은 0.5%였습니다.
- petition과 wiki에서도 DPO의 ASR이 향상되었으나 acute 비율이 함께 증가했으며, 특히 wiki는 9.4%로 나타났습니다.

### Repetition diagnostics

문장 재사용과 근사 중복은 문장쌍 1,000개당 발생량으로 보고합니다. 도메인마다 인간 문서의
기준선이 다르므로 도메인 간 절댓값은 직접 비교하지 않습니다.

| domain | human reuse | DPO reuse | human approximate | DPO approximate |
| --- | ---: | ---: | ---: | ---: |
| essay | 0.42 | 1.51 | 14.29 | 20.73 |
| persona | 0.87 | 1.09 | 125.34 | 123.15 |
| petition | 1.14 | 13.92 | 43.32 | 50.97 |
| wiki | 4.25 | 12.58 | 44.44 | 43.30 |

### Attribution and release boundary

이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된
AI Hub의 「에세이 글 평가 데이터」와 「페르소나 대화」를 활용했습니다. 국민청원 데이터는
Hugging Face의 `dev7halo/bluehouse-national-petition`(Apache-2.0)을, 위키 데이터는
Wikimedia의 한국어 위키백과 덤프(CC BY-SA)를 활용했습니다.

공개 범위는 집계 통계와 평가 코드로 제한합니다. 인간 원문, AI 재서술문, 프롬프트,
문서별 점수, 데이터 분할 식별자, 모델 체크포인트와 검수 자료는 배포하지 않습니다.

## Research records

실험 결과, checkpoint 계보, 도메인별 분석과 실패 기록은 [notes/README.md](notes/README.md)와
[progress.md](progress.md)에 있습니다. 데이터 명세는 [mash/DATASET.md](mash/DATASET.md),
별도 반복학습 실험은 [loop/README.md](loop/README.md)를 참고합니다.

방법론은 MASH와 Russell et al.의 text-humanization 및 탐지 회피 평가 설정을 바탕으로 합니다.
