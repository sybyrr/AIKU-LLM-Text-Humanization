# AIKU Korean LLM Text Humanization

한국어 AI 생성문을 사람 문체로 재서술하는 StyleBART 기반 연구 파이프라인이다.
도메인별 탐지기로 학습 pair를 구성하고 SFT와 DPOP를 거친 뒤 held-out test에서
탐지 회피와 반복 붕괴를 평가한다.

> 이 저장소에는 코드와 문서만 포함한다. 인간 원문, 생성문, checkpoint, 평가 로그와
> 검수 HTML은 저작권·데이터 이용약관 때문에 배포하지 않는다.

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

현재 정본은 [pipeline/README.md](pipeline/README.md)에 정의되어 있다.
`loop/`는 generator와 detector를 반복 재학습하는 별도 실험이며 위 정본 파이프라인과
설정이나 결과를 합치지 않는다.

## Repository layout

```text
pipeline/                 정본 설정, 오케스트레이터, 최종평가, 반복 지표
pipeline/configs/         공개 가능한 도메인 설정 예시
scripts/                  정본 단계 구현과 과거 연구용 도구
loop/                     별도 generator↔detector 반복학습 실험
notes/                    실험 설계·결과·시행착오 기록
mash/                     데이터셋 및 프롬프트 명세
```

정본 단계 구현은 다음 파일이다.

- `scripts/build_domain_prompts.py`
- `scripts/generate.py`
- `scripts/domain_gate.py`
- `scripts/stage2_sft.py`
- `scripts/stage3_build_dpo.py`
- `scripts/stage3_dpo.py`
- `pipeline/evaluate.py`

그 밖의 t-SNE, cross-domain, LLM judge, HTML 검수 코드는 분석·실험 도구이며 정본
학습 경로에 자동으로 들어가지 않는다.

## Environment

Python 3.10 이상과 CUDA 환경이 필요하다. GPU에 맞는 PyTorch를 먼저 설치한 뒤:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

생성 단계는 OpenAI-compatible `/v1/chat/completions` endpoint를 사용한다.
Qwen과 EXAONE 서버 주소는 설정 파일에 각각 지정한다.

## Input

도메인별 `human_pool.jsonl`의 최소 스키마는 다음과 같다.

```json
{"doc_id":"unique-id","text":"사람이 작성한 원문","n_char":934}
```

본문은 `text`, `body`, `human_text` 중 하나를 사용할 수 있다. 동일 논제·도서·대화가
train/test에 함께 들어가면 안 되는 도메인은 그룹 필드를 추가하고 `gate.split_key`로
지정한다. 대화 데이터는 `n_turn`과 `gate.dialogue=true`를 사용한다.

## Run

신문 플래그십 설정을 복사해 비공개 경로와 생성 서버 주소를 수정한다.

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

개별 단계만 실행할 수 있다.

```bash
python pipeline/run.py \
  --config pipeline/config.news.json \
  --gpu 0 \
  --from-stage sft \
  --to-stage evaluate
```

기본 실행은 완료 산출물을 검증하고 이어서 실행한다. 로그와 상태 파일은
`logs/{domain}/pipeline/`에 기록된다. 로컬 설정, 데이터, 모델과 로그는 `.gitignore`
대상이다.

## Canonical settings

| 단계 | 설정 |
| --- | --- |
| SFT | ko-BART, concat→projection fusion, target EOS, max length 512, λ=0.5, 3 epochs |
| Hard negative | SFT sampling, D score > τ, 4 candidates, 기존 재현값 `no_repeat_ngram_size=3` |
| DPO | length normalization, β=2, DPOP λ=5, 1 epoch, effective batch 16 |
| Final generation | plain beam4, input max 512, output max 1024, 반복방지 옵션 없음 |

`hard_negative.no_repeat_ngram_size=0`은 붕괴 출력도 rejected 후보로 노출하는 별도
실험 조건이다. 기존 결과를 재현할 때는 기본값 3을 유지한다.

## Evaluation

최종평가는 pair 파일의 test split만 사용한다.

- 주 지표: 도메인 인간 원문 P(AI)의 95분위 임계에서 raw ASR
- 안전성: 공백 제거 문자 6-gram이 20회를 초과해 반복되는 급성붕괴율
- 의미 보존: ko-sroberta cosine
- 선택 평가: 동일 detector split으로 학습한 SCRN

collapse는 raw ASR에서 차감하지 않고 별도로 보고한다. LLM-as-judge 품질 점수는
모델별 편차가 커 정본 평가에서 제외한다. 평가기는 문서별 JSONL과 집계
`*.summary.json`을 함께 저장한다.

## CPU checks

GPU를 쓰지 않고 설정과 command graph, 재개 판정, 표본 추출 및 반복 지표를 검사한다.

```bash
python -m compileall -q pipeline scripts
python -m unittest discover -s pipeline/tests -v
python pipeline/run.py --config pipeline/configs/news.example.json --dry-run
```

## Research records

실험 결과, checkpoint 계보, 도메인별 분석과 실패 기록은 [notes/README.md](notes/README.md)와
[progress.md](progress.md)에 있다. 데이터 명세는 [mash/DATASET.md](mash/DATASET.md),
별도 반복학습 실험은 [loop/README.md](loop/README.md)를 참고한다.

방법론은 MASH와 Russell et al.의 text-humanization 및 탐지 회피 평가 설정을 바탕으로 한다.
