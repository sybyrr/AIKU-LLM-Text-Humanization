# Canonical humanization pipeline

이 디렉터리는 실험별 셸 스크립트가 아니라, 새 도메인에도 반복 적용할 수 있는 현재 정본
파이프라인의 진입점입니다. 실제 데이터 처리·학습 구현은 `../scripts/`에 있고 여기서는 설정,
단계 연결, 재개, 산출물 검증을 담당합니다.

## 범위

```text
human_pool.jsonl
  → P3b 공통 프롬프트
  → 복수 LLM 재서술
  → frozen domain detector gate
  → StyleBART SFT (concat fusion + EOS)
  → SFT hard-negative 채굴
  → DPOP
  → held-out test plain-beam 평가
```

도메인마다 달라지는 것은 `human_pool.jsonl`을 만드는 Stage 1a와 설정값뿐입니다. 데이터,
생성물, 모델, 로그는 공개 저장소에 넣지 않습니다.

## 빠른 시작

```bash
cp pipeline/configs/news.example.json pipeline/config.news.json
# domain, paths, generators, gate의 split_key/dialogue 등을 수정

python pipeline/run.py \
  --config pipeline/config.news.json \
  --workspace /path/to/private-workspace \
  --gpu 0 \
  --dry-run

python pipeline/run.py \
  --config pipeline/config.news.json \
  --workspace /path/to/private-workspace \
  --gpu 0
```

`workspace` 아래에 `dataset/`, `models/`, `logs/`가 생깁니다. 저장소 안에 비공개 데이터를
둘 경우에는 `--workspace .`을 사용합니다. 각 생성기 URL의 llama-server는 실행 전에 별도로
기동해야 합니다.

## 입력 계약

### `human_pool.jsonl`

필수 필드는 `doc_id`와 본문 하나(`text`, `body`, `human_text`)입니다.

```json
{"doc_id":"unique-id","text":"사람이 작성한 원문","n_char":934}
```

그룹 단위 분할이 필요하면 `prompt`, `book_id` 같은 그룹 필드도 넣고 `gate.split_key`에
그 이름을 지정합니다. 대화 도메인은 `n_turn`을 넣고 `gate.dialogue=true`로 둡니다.

### 생성기

`generators` 배열의 항목마다 서로 다른 OpenAI-compatible chat endpoint를 지정합니다. 생성은
항목 순서대로 수행되며 각 결과는 `name=JSONL` arm으로 gate에 모두 전달됩니다. Qwen3 계열은
`no_think=true`가 필요합니다.

## 단계와 재개

| 단계 | 구현 | 완료 판정 |
| --- | --- | --- |
| `prompts` | `build_domain_prompts.py` | human pool과 같은 수의 prompt |
| `generate` | `generate.py --resume` | 모든 `(doc_id, cond)` 성공 출력 존재 |
| `gate` | `domain_gate.py` | 최소 pair 수 + detector config 존재 |
| `sft` | `stage2_sft.py --resume` | `style_bart.pt` 존재 |
| `hard-negative` | `stage3_build_dpo.py` | 전량 처리 summary marker 존재 |
| `dpo` | `stage3_dpo.py` | `dpo_bart.pt` 존재 |
| `evaluate` | `pipeline/evaluate.py` | 평가 JSONL + summary JSON 존재 |

기본 실행은 완료 산출물을 건너뜁니다. 범위를 제한하려면 `--from-stage`와 `--to-stage`를 씁니다.
`--force`는 완료 판정을 무시하고 하위 단계를 다시 호출하지만, `generate.py`와 hard-negative
채굴처럼 자체 재개 기능이 있는 구현은 기존 성공분을 재사용합니다. 완전 초기화는 산출물을 별도로
백업한 뒤 명시적으로 치워야 합니다. 단계별 stdout/stderr와 상태는
`logs/{domain}/pipeline/`에 기록됩니다.

## 정본 설정

- SFT: ko-BART, concat→projection fusion, target EOS, `max_length=512`, λ=0.5, 3 epochs.
- DPO: DPOP λ=5, length-normalized β=2, 1 epoch, effective batch 16.
- 최종 생성: plain beam4, 입력 512, 출력 상한 1024, 반복방지 옵션 없음.
- 주 지표: target-domain human P(AI)의 95분위 임계에서 raw ASR.
- 안전성: 급성붕괴율을 raw ASR과 별도로 보고.
- 의미 보존: ko-sroberta cosine을 별도로 보고. LLM-as-judge 품질 점수는 정본에 포함하지 않음.

`hard_negative.no_repeat_ngram_size`의 기본값 3은 기존 채굴 조건을 재현합니다. 최종 plain
평가는 반복방지를 사용하지 않으므로 두 분포가 다릅니다. 붕괴 출력도 rejected 후보로 노출하는
실험을 하려면 이 값을 0으로 명시하되, 기존 결과와 같은 실험이라고 취급하면 안 됩니다.

## 평가 출력

`evaluate.py`는 test split에서 `human`, `x_ai`, `SFT`, `DPO` 네 행을 문서별로 저장합니다.
표본 평가에서는 generator 비율을 largest-remainder 방식으로 보존하고 seed 42로 섞습니다.
`evaluation.sample_size=0`은 test 전량을 뜻합니다.

각 JSONL 행에는 다음이 들어갑니다.

```text
domain, dec, doc_id, generator, text,
p_D, [p_SCRN], cos, rep_reuse, rep_approx, rep_acute
```

행 단위 파일과 함께 `<이름>.summary.json`에 프로토콜, 생성기 구성, 탐지기 분리도,
모집단별 raw ASR·collapse·cosine·반복 진단 요약을 저장합니다.
평가기는 같은 실행에서 생성한 human 점수의 95분위를 임계로 사용합니다. 서로 다른 target
detector의 ASR 절댓값을 직접 비교하기보다 같은 target 안에서 SFT→DPO를 비교합니다.

## CPU 검증

GPU 없이 설정, 단계 연결, 층화 표본, 붕괴 판정을 검사할 수 있습니다.

```bash
python -m unittest discover -s pipeline/tests -v
python pipeline/run.py --config pipeline/configs/news.example.json --dry-run
```
