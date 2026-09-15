# Main humanization pipeline

한국어 Human–AI pair 구축부터 StyleBART SFT/DPOP 학습과 held-out 평가까지 실행합니다.
단계별 구현은 [`scripts/`](../scripts/README.md), 데이터 준비는
[실행 가이드](../docs/reproduction.md), 모델·지표 정의는 [방법론](../docs/methodology.md)을
참고하세요.

```text
human_pool.jsonl
  → P3b inverse rewriting prompt
  → Qwen / EXAONE rewrite
  → frozen RoBERTa gate
  → StyleBART SFT
  → SFT hard negatives
  → DPOP
  → held-out test evaluation
```

## 실행

```bash
cp pipeline/configs/news.example.json pipeline/configs/news.local.json

python pipeline/run.py \
  --config pipeline/configs/news.local.json \
  --workspace /path/to/workspace \
  --gpu 0 --dry-run

python pipeline/run.py \
  --config pipeline/configs/news.local.json \
  --workspace /path/to/workspace \
  --gpu 0
```

`--dry-run`은 단계별 명령을 출력합니다. `workspace` 아래에는 `dataset/`, `models/`,
`logs/`를 배치하며 생성기별 llama-server를 먼저 실행합니다.

상대 산출물 경로는 workspace 기준입니다. Config의 상대 `workspace`는 config 파일이
있는 디렉터리를 기준으로 해석하므로, 예시 config는 같은 디렉터리에 복사하거나
`--workspace`를 명시합니다.

여러 gate 결과를 기사당 한 pair로 병합할 때는
[`build_clean_dataset.py`](../scripts/build_clean_dataset.py)를 별도로 실행합니다.
최종 pair와 detector가 있으면 config에 경로를 지정하고 `--from-stage sft`,
checkpoint 평가만 실행하면 `--from-stage evaluate`를 사용합니다.

## 입력과 설정

`human_pool.jsonl`에는 `doc_id`와 본문 필드(`text`, `body`, `human_text` 중 하나)가 필요합니다.

```json
{"doc_id":"unique-id","text":"사람이 작성한 원문","n_char":934}
```

그룹 단위 분할은 `prompt`, `book_id` 등의 필드를 추가하고 `gate.split_key`에 지정합니다.
대화 데이터는 `n_turn`을 넣고 `gate.dialogue=true`로 설정합니다.

`generators`에는 모델별 OpenAI-compatible llama-server endpoint를 지정합니다.
생성기 순서대로 결과를 만들고 모두 gate에 전달합니다. Qwen3은 `no_think=true`를
사용합니다. `model`은 결과에 기록할 이름이며, 실제 모델은 endpoint에서 선택합니다.

## 단계와 재개

| 단계 | 구현 | 완료 판정 |
| --- | --- | --- |
| `prompts` | `build_domain_prompts.py` | human pool과 같은 수의 prompt |
| `generate` | `generate.py --resume` | 모든 `(doc_id, cond)` 성공 출력 |
| `gate` | `domain_gate.py` | 최소 pair 수 + detector config |
| `sft` | `stage2_sft.py --resume` | `style_bart.pt` |
| `hard-negative` | `stage3_build_dpo.py` | pair + 처리 완료 summary |
| `dpo` | `stage3_dpo.py` | `dpo_bart.pt` |
| `evaluate` | `pipeline/evaluate.py` | 평가 JSONL + summary |

완료된 산출물은 건너뜁니다. `--from-stage`, `--to-stage`로 실행 범위를 지정하며,
로그와 상태는 `logs/{domain}/pipeline/`에 저장됩니다. 새로운 조건에는 새 출력 경로를
사용합니다. `--force`는 각 스크립트를 다시 호출하며, 스크립트 내부 재개·캐시는 유지합니다.

## 기본 실험 조건

| 항목 | 설정 |
| --- | --- |
| Generator | `gogamza/kobart-base-v2`, concat → shared projection |
| SFT | target EOS, 입력/정답 최대 512, λ=0.5, 3 epochs |
| DPOP | λ=5, token-average log probability, β=2, 1 epoch, 유효 batch 16 |
| Hard negative | 입력당 후보 4개, RoBERTa `P(AI)>0.5`, 최대 점수 후보 |
| 평가 생성 | plain beam 4, 입력 최대 512, 출력 최대 1,024, 반복 억제 없음 |
| 주 지표 | 평가 대상 Human 점수의 P95 임계값에서 raw ASR |
| 내용 유사도 | `jhgan/ko-sroberta-multitask` cosine, 최대 256토큰 |
| 붕괴 | 공백 제거 문자 6-gram이 20회를 초과해 반복되는 출력의 비율 |

Hard-negative 채굴은 `no_repeat_ngram_size=3`, 최종 평가는 반복 억제 없이 실행합니다.
붕괴율은 raw ASR와 별도로 보고하며, generator 출력을 추가 교정 없이 평가합니다.

SCRN은 `evaluation.scrn`에 `scrn_best.pt` 경로를 지정하면 추가됩니다. 기본값은 `null`이며
RoBERTa만 평가합니다. 학습 입력과 checkpoint 준비는
[SCRN 실행 안내](../docs/reproduction.md#scrn-평가)를 참고하세요.

## 평가 출력

문서별 `human`, `x_ai`, `SFT`, `DPO` 네 버전을 저장합니다.
`evaluation.sample_size=0`이면 test 전량을 사용합니다. 표본 평가에서는 largest-remainder
할당으로 생성기 비율을 보존하고 seed 42로 표본을 추출합니다.

```text
domain, dec, doc_id, generator, text,
p_D, [p_SCRN], cos, rep_reuse, rep_approx, rep_acute
```

`*.summary.json`에는 프로토콜, 생성기 구성, detector 임계값과
raw ASR·붕괴·cosine·반복 집계를 기록합니다.
