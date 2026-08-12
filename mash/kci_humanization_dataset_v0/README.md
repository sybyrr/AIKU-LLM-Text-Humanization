# KCI Korean Humanization Dataset v0

## Status

- Human corpus 구축 완료
- synthetic AI-Human pair generation pipeline 구현 완료
- EXAONE + prompt v3 pilot 검증 중
- 전체 1,497 pair generation은 아직 수행하지 않음

이 패키지는 **Human-only v0** release입니다. AI-Human pair JSONL은 포함하지 않습니다.

## Overview

한국어 LLM humanization 연구를 위한 KCI 논문 초록 기반 Human corpus입니다.

- **Human text**는 실제 KCI 논문 초록입니다.
- **AI text**는 Human abstract를 `LGAI-EXAONE/EXAONE-4.0-1.2B` + prompt v3로 faithful paraphrase할 예정입니다.
- 전체 1,497건 pair generation은 아직 수행하지 않았습니다.

## Human Corpus Construction

KCI OAI-PMH raw collection: **5,000** records

이후 다음 조건으로 cleaning했습니다.

- metadata `language == "한국어"`
- actual `original_abstract` Hangul ratio ≥ 0.30
- abstract length ≥ 200 chars
- publication year ≤ 2021
- duplicate ID 제거
- duplicate abstract 제거

clean Korean corpus: **2,194** records

특정 journal 문체 편향 완화를 위해 journal당 최대 100건을 `seed=42`로 sampling했습니다.

최종 Human corpus: **1,497** records

KCI metadata의 `language`가 한국어여도 영어 또는 중국어 초록이 존재하는 사례가 확인되어, metadata뿐 아니라 실제 abstract의 Hangul ratio 검증을 추가했습니다.

## Pair Generation

synthetic AI-Human pair generation pipeline은 구현되어 있습니다.

예정 설정:

- Generator: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- Prompt version: v3
- do_sample: false
- enable_thinking: false

Prompt v3는 sentence-by-sentence faithful paraphrase 방식이며 다음을 목표로 합니다.

- 요약 최소화
- 원문의 문장/정보 보존
- 숫자/연도/비율/금액/단위 보존
- 증가/감소, 정(+)/부(-), 유의/비유의 등의 방향 보존
- 새로운 해석/사실 추가 금지

현재는 EXAONE + prompt v3 **pilot 검증 중**이며, 전체 1,497 pair generation은 아직 수행하지 않았습니다.

실제 prompt는 `prompt_v3.txt`를 참고하세요.

## Data Schema

Human corpus 주요 field:

- `kci_article_id`
- `original_title`
- `original_abstract`
- `research_field`
- `journal`
- `publisher`
- `publication_date`
- `language`
- `kci_url`
- `abstract_hangul_ratio`

Pair 주요 field (향후 pair release에 포함 예정):

- `id`
- `human_text`
- `ai_text`
- `generator`
- `prompt_version`
- `research_field`
- `journal`
- `publication_date`
- `original_title`
- `publisher`
- `kci_url`
- `generation_config`

## Dataset Statistics

숫자는 이 release를 만들 때 실제 Human corpus 파일을 읽어 계산했습니다.

| Metric | Value |
| --- | --- |
| Human records | 1,497 |
| Pairs | not generated |
| Research fields | 67 |
| Journals | 167 |
| Publication years | 1996–2020 |
| Human mean length (chars) | 642.62 |
| Human median length (chars) | 607.00 |

## Files

| File | Role |
| --- | --- |
| `kci_human_ko_cap100.jsonl` | 최종 Human corpus (journal-cap 100, Korean-text) |
| `README.md` | 출처, 구축 과정, 스키마, 통계 |
| `prompt_v3.txt` | 예정 pair generation에 사용할 prompt v3 |
| `dataset_stats.json` | 기계가 읽기 쉬운 Human corpus 통계 |

## Limitations

이 데이터는 초기 **v0 Human corpus**입니다. 전체 synthetic pair는 아직 포함하지 않습니다.

Human corpus는 실제 한국어 논문 초록을 대상으로 정제했으나, 이후 AI paraphrase에는 일부 정보 축약, 어색한 표현, 의미 관계 변화, factual inconsistency가 존재할 수 있습니다.

이 패키지를 검증 완료된 paired dataset으로 간주하지 마세요.

Release directory: `kci_humanization_dataset_v0`
