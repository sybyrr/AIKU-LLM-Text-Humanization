# Scripts

주 실행기는 [`pipeline/run.py`](../pipeline/run.py)입니다.
입력 준비와 명령은 [실행 가이드](../docs/reproduction.md), 데이터 구성은
[데이터 안내](../docs/data.md), 모델·지표 정의는 [방법론](../docs/methodology.md)에 있습니다.

## 데이터 구축과 학습

| 파일 | 역할 |
| --- | --- |
| `download_corpus.py` | 국립국어원 말뭉치 다운로드 보조 |
| `mash_extract_pool.py` | 신문 2022판에서 2021년 인간 기사 추출, `--corpus`로 ZIP 지정 |
| `build_domain_prompts.py` | 전 도메인 공통 P3b inverse rewriting prompt |
| `generate.py` | llama-server 병렬 생성·재개 |
| `domain_gate.py` | RoBERTa 학습 문서 분리·동결·Human/AI 후보 선별·분할 |
| `build_clean_dataset.py` | 선별 결과 병합·D 학습 문서 제외·기사 중복 제거·재분할 |
| `stage2_sft.py` | 공유 fusion과 AI/Human 스타일 벡터를 쓰는 StyleBART SFT |
| `stage3_build_dpo.py` | SFT 후보 중 RoBERTa가 잡는 hard negative 채굴 |
| `stage3_dpo.py` | DPO/DPOP preference 학습 |

`stage3_dpo.py`의 기본값은 vanilla DPO입니다. DPOP 실험은
`--dpop --length-norm --beta 2`를 사용하며, 전체 설정은
[news config](../pipeline/configs/news.example.json)에 있습니다.

## 평가와 분석

| 파일 | 역할 |
| --- | --- |
| `../pipeline/evaluate.py` | held-out 생성·RoBERTa/SCRN·cosine·반복 평가 |
| `../pipeline/metrics_redundancy.py` | 문자 6-gram 붕괴·문장 재사용·근사중복 지표 |
| `scrn_train.py`, `scrn_score.py` | KoELECTRA 기반 SCRN detector 학습·채점 |
| `analyze_domain.py` | RoBERTa CLS 표현의 t-SNE; test 앞 N개 사용 |
| `audit_overlap.py` | pair와 D 학습·generator train/test 문서 겹침 확인 |
| `eval_collapse.py` | checkpoint별 반복 붕괴 진단 |
| `metrics_quality.py` | 별도 BERTScore-F1 분석 |
| `make_stage2_review.py`, `make_dpo_review.py` | 원문·생성문 비교 HTML 작성 |
| `figures/` | 집계 결과의 그래프 재생성 |

## Prompting baseline과 실행 환경

| 파일 | 역할 |
| --- | --- |
| `build_humanizer_skill_baseline.py` | 동일 held-out 입력·prompt shard 준비 |
| `run_humanizer_skill_baseline.py` | 20편 파일럿 후 100편 비교 실행 |
| `eval_humanizer_skill_baseline.py` | baseline의 RoBERTa/SCRN·cosine·반복 평가 |
| `summarize_humanizer_skill_baseline.py` | matched baseline 비교·품질 게이트 집계 |
| `gpu_serve.sh` | llama-server 기동 보조; 실행 환경의 경로·GPU 옵션 설정 |

Baseline prompt와 설정은 [`baselines/`](../baselines/README.md),
D1 적응·G2 추가 학습과 CopyKiller 평가는 [`adversarial/`](../adversarial/README.md)에 있습니다.
