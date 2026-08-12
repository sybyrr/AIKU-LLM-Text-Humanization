# CLAUDE.md

한국어 LLM 생성문의 "AI 티" 억제 연구 워크스페이스. 두 방식으로 인간↔AI pair 데이터를 수집한다:

- **Russell (ACL 2025) 트랙** — 사람 신문기사 100편 ↔ 조건별 AI 생성문 (이 폴더의 기존 파이프라인)
- **MASH 트랙** — KCI 논문 초록 1,497편 기반, 팀원(김용재)에게서 v0 인수 (`mash/`; 배경은 [notes/30](notes/30-MASH-기반-제안.md)과 [archive/notion-export의 0803-진민재 노트](archive/notion-export/))

1차 목표는 두 데이터셋 모두 탐지기(카피킬러 + Pangram 예정)에 넣어 AI 글→AI, 인간 글→인간으로 판별되는지 확인해 **데이터 완전성을 검증**하는 것. 이후 MASH 방식으로 LLM 튜닝 예정.

## 문서 역할 분담 (중복 금지)

같은 정보는 **한 곳에만** 두고 나머지는 링크한다.

| 파일 | 담당 | 갱신 시점 |
| --- | --- | --- |
| **CLAUDE.md** (이 파일) | 규칙 · 폴더 지도 · 용어 | 구조/규칙이 바뀔 때만 |
| [progress.md](progress.md) | 진행 현황 · 다음 할 일 · 알려진 문제 | 상태가 바뀔 때마다 (유일한 상태 기록처) |
| [notes/README.md](notes/README.md) | 연구 노트 색인 | 노트가 추가될 때 |

연구 배경·의사결정 근거·실험 수치는 notes/ 안에만 있다 (실행 기록의 정본은 [notes/42](notes/42-생성-명세와-결과.md)).

## 폴더 지도

| 경로 | 내용 |
| --- | --- |
| `dataset/human/` | 인간 풀 100편 — 정본은 `full_base.jsonl` (russell_pool_100.\* 은 같은 100편의 원형/CSV) |
| `dataset/prompts/` | 조건 A/C/E 프롬프트 300개(`prompts_full.jsonl`) + 재료(요약·명사) |
| `dataset/generations/` | 모델별 300건 × 8모델 + `full_generations.jsonl`(8모델 합본 2,400건; **quant/ 미포함**, 오류 레코드 1건 포함) |
| `dataset/generations/quant/` | qwen3-8b Q4 vs Q8 비교 ablation (조건 A만, 합본·점수 미포함) |
| `dataset/scores/` | 탐지기 점수 (현재 Fast-DetectGPT만; **구세트 대상** — progress.md 참조) |
| `mash/` | **MASH 트랙 본진.** 인간 풀 `human_pool.jsonl`(9,425편) ← 인수분 v0 1,497 + API 수집 v1 8,000. 확정 프롬프트 `prompt_p3b.md`, 생성 결과 `gen_full_p3b.jsonl`. AI Hub 판정·코퍼스 조사 문서 동봉 |
| `pilot/` | Stage 0 파일럿 산출물 — `data/`(시드·프롬프트) `gen/`(생성) `qa/`(복사·보존 검사) `scores/`(카피킬러 점수 + `copykiller/` 결과확인서 PDF 원본) `export/`(탐지기 업로드용 docx·zip) |
| `corpus/raw/` | NIKL 공식 배포 zip 2개 (gitignore, 1.8GB) |
| `corpus/mirror/` | HF 미러 parquet 자리 — **이 사본엔 없음** (안내: `corpus/mirror/README.md`) |
| `models/` | GGUF 가중치 (gitignore; 이 사본엔 EXAONE-3.5-7.8B만 복사됨) |
| `scripts/` | 파이프라인 전체 (아래 실행 흐름) |
| `review/` | 사람 검수용 정적 HTML 3종 + full_cards.json |
| `logs/` | 변환·양자화·서빙·수집·생성 실행 로그 — 증거 보존용, **로컬 전용**(gitignore) |
| `notes/` | 연구 노트 15편 (진입점: [notes/README.md](notes/README.md)) |
| `archive/` | 파일럿 잔여물 · 구버전 스냅샷 · Notion 원본 export · `aihub_essay_2024_raw/`(AI Hub 논증문 원본, [사용 불가 판정](mash/aihub_verdict.md)의 근거물) · `handoff_20260811.md`(인수 시점 기획, progress.md § M 으로 대체됨) · `claude-home-backup/`(**이메일·대화 전문 포함 — 커밋 금지**) — **읽기 전용, 삭제 금지** |
| `team_brief.md` | 팀원 공유용 요약 (루트) — 상세는 progress.md·notes/ |

## 스크립트 실행 흐름

```
download_corpus.py                          # NIKL API → corpus/raw
→ extract_russell_pool.sql                  # (duckdb) corpus/mirror → 인간 풀 100편
→ summarize.py(조건C) · extract_nouns.py(조건E)
→ 프롬프트 조립 SQL (run_full.sh 내장; build_prompts_AEC.sql은 파일럿 스케일 버전)
→ generate.py  ← 러너: run_pilot / run_full / run_night / run_domestic.sh (llama-server 8080/8081)
→ check_copying.py (복사 유출 QA)
→ detect_fastdetectgpt.py · detect_llm_judge.py (+ prompts/russell/ 템플릿 5종)
→ make_*_html.sh → review/
```

## 용어 · 고정 설정

- 조건 **A**=제목만 · **C**=제목+요약 1문장 · **E**=제목+핵심명사 25개 · ~~B~~=제목+도입부 2문장(복사 유출로 폐기) · **A2**=A의 페르소나 문구 리라이트(정보 조건 아님) · **t0**=temperature 0 프로브
- 생성: temperature 0.0, Q4_K_M 통일, 마크다운·제목 재출력 금지, max_tokens=목표자수×0.8
- Fast-DetectGPT 임계값 0.126 (인간 FPR 5%), 점수는 mean 방식 (sum 방식은 버그 — archive/detect_scores_buggy.jsonl)

## 규칙

1. **새 폴더/파일명은 영어로.** 예외: notes/*.md 한글 파일명과 archive/notion-export/ 내부는 상호 링크 보존을 위해 이름 변경 금지.
2. **문서는 참조 우선.** 다른 파일에 있는 내용을 복사하지 말고 링크한다. 상태 변화는 progress.md 한 곳만 갱신한다.
3. **비밀 보호.** `.kli_api_key`(국립국어원 API 키)는 열람·출력·커밋 금지. gitignore 대상: `.kli_api_key`, `.venv/`, `models/`, `corpus/raw/`.
4. **archive/ 는 지우지 않는다.** 조건 B 폐기 근거, 기존 점수의 출처인 9모델 스냅샷 등 증거가 들어 있다.
5. **dataset/ 의 경로·파일명은 scripts/ 하드코딩 기본값과 결합**되어 있다 — 이동·개명 전에 반드시 `grep -r` 로 참조를 확인한다.
6. **이 사본에서 파이프라인 재실행을 가정하지 말 것.** 모델 GGUF 대부분·llama.cpp·duckdb 부재, .venv 심링크 깨짐 → 산출물 아카이브로 취급 (자세히: [progress.md § 알려진 문제](progress.md)).
7. **에이전트 작업 배분:** 파일 읽기·코드 실행처럼 단순하지만 양이 많은 작업은 opus 서브에이전트에, 고수준 추론·다음 단계 제안은 fable(메인)이 맡는다.
