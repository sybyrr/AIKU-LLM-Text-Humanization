# scripts/ — 파이프라인 스크립트 지도

정본 실행: [../pipeline/README.md](../pipeline/README.md) · 연구 기록: [../notes/README.md](../notes/README.md)

> 새 실행은 [../pipeline/run.py](../pipeline/README.md)를 진입점으로 사용합니다. 이 디렉터리는
> 단계별 구현체와 연구용 분석 도구를 보관하며, 정본 설정·재개·평가는 pipeline/에서 관리합니다.

데이터·모델은 저장소에 없다(라이선스 — 루트 README 참조). 아래 스크립트는 `dataset/`·`models/` 를
로컬에 배치한 상태에서 돕니다.

## Stage 1–3 정본 파이프라인 (도메인 무관)

| 순서 | 스크립트 | 역할 |
| --- | --- | --- |
| 1a | `mash_extract_pool.py` | (신문) 인간 원문 추출 → `human_pool.jsonl`. **도메인마다 새로 작성**하는 유일한 단계 |
| 1b | `build_domain_prompts.py` | P3b 통일 프롬프트 조립 (전 도메인 공용, `rewrite_ko` 대체) |
| 1b | `generate.py` | llama-server 병렬 생성 (`--resume`) → x_ai |
| 1c | `domain_gate.py` | 도메인 탐지기 D 동결 학습 + 게이트(d_human<τ AND d_ai≥τ). `--frozen-only` 제외 가드 포함 |
| 1d | `build_clean_dataset.py` | 게이트 여러 번(코어·확장·top-up) 합칠 때: D학습 제외·문서 dedup·분할·누수 0 검증 (선택) |
| 2 | `stage2_sft.py` | StyleBART(concat fusion + EOS) SFT, 이중경로 λ0.5 |
| 3 | `stage3_build_dpo.py` → `stage3_dpo.py` | hard-neg 채굴 → **DPOP λ5** (플래그 명시 필수 — 기본값은 vanilla) |
| legacy | `news_gate_frozen.py` | `domain_gate.py`의 과거 도메인 전용 파일명 |
| legacy | `run_domain_pipeline.sh` | 과거 셸 오케스트레이터. 새 실행은 `pipeline/run.py` 사용 |

## 평가 · 시각화

| 스크립트 | 역할 |
| --- | --- |
| `../pipeline/evaluate.py` | 정본 held-out 평가: raw ASR·collapse·cosine·반복 진단 |
| `../pipeline/metrics_redundancy.py` | 정본 급성붕괴·문장 재사용·근사중복 정의 |
| `eval_collapse.py` | 과거 checkpoint 단독 붕괴 분석 |
| `analyze_domain.py` | roberta-D CLS 특징 t-SNE + D/SCRN P(AI) |
| `scrn_train.py` / `scrn_score.py` | 도메인별 독립 탐지기 SCRN(koelectra) 학습·채점 |
| `detect_binoculars.py` / `detect_fastdetectgpt.py` / `detect_llm_judge.py` | 전이(zero-shot·LLM) 탐지기 |
| `metrics_quality.py` | BERTScore-F1 의미 보존 |
| `audit_overlap.py` | 누수 감사 (∩D학습=0, train∩test=0) |
| `make_dpo_review.py` / `make_stage2_review.py` | SFT/DPO 검수 HTML |

## 초록(KCI) 트랙 — 별도 도메인 구축

`collect_kci*.py`, `stage0_*.py`, `build_human_pool.py`, `build_final_dataset.py` 는 초록(KCI 논문 초록)
도메인의 수집·검사·조립이다(카피킬러 게이트 기반, 별도 명세는 `../mash/PIPELINE.md` 참고). Stage 2–3
학습은 위 정본 파이프라인과 공유합니다.

## 서버 · 유틸

`gpu_serve.sh`(llama-server 기동), `run_pilot.sh` 등은 생성 서버·파일럿 실행 보조입니다.
