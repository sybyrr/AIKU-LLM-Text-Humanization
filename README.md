# 한국어 LLM 생성문의 "AI 티" 억제 — Style Humanization 파이프라인

한국어로 쓴 글이 AI 탐지기에 어떻게 걸리는지 재고, **인간 글 ↔ AI 글 쌍**을 만들어
그 데이터로 **문체 변환 모델(humanizer)**을 학습한다. AI가 쓴 한국어 글을 humanizer로 다시 써서,
내용·품질을 유지한 채 탐지기가 "사람 글"로 오판하게 만들 수 있는지 — 나아가 학습에 안 쓴
탐지기까지 속이는지(전이) 검증한다.

고려대 학부 프로젝트(AIKU)이며, 방법론은
[MASH (ACL 2026 Findings)](https://arxiv.org/abs/2601.08564)와 Russell et al. (ACL 2025)를 따른다.

> **이 저장소에는 코드와 문서만 있다.** 인간 원문·생성 pair·프롬프트·점수·모델·검수 HTML 은
> 저작권·약관(국립국어원 모두의 말뭉치 제12조, KCI 초록 저작권, AI Hub 제3자 열람 금지) 때문에
> 제외했고, `.gitignore` 가 이 경계를 강제한다. 데이터는 참여자에게 Google Drive 로만 공유한다.

---

## 지금까지 (한눈에)

| 도메인 | 인간 소스 | 규모 | 상태 |
| --- | --- | --- | --- |
| **신문** | 국립국어원 신문 말뭉치 clean20k | **pair 20,343** (train 16,266 / dev 2,033 / test 2,044) | **Stage 1–3 완료** — 플래그십 |
| 초록(메인) | KCI 논문 초록 | pair 5,553 (인간 9,425) | 탐지기 검증 완료(카피킬러 F1 0.959) |
| 청원 | 국민청원(HF `dev7halo/bluehouse…`) | pair 1,425 (인간 8,000) | 파일럿 — 일반화 검증 |
| 위키 | 위키백과(HF `wikimedia/wikipedia ko`) | pair 1,197 (인간 8,000) | 파일럿 |
| 에세이(논증문) | AI Hub 에세이 평가 데이터 `dataSetSn=545`(2021, 50,413편) | — | 신청·추출 예정 |

**신문 결과(clean20k test held-out, n=80, plain beam4)** — 탐지기 P(AI), 낮을수록 사람:

| 단계 | roberta-D | SCRN(전이) |
| --- | --- | --- |
| 인간 원문 | 0.04 | 0.00 |
| x_ai (AI 재서술) | 0.999 | 1.00 |
| **SFT** | 0.23 | 0.16 |
| **DPO (DPOP λ5)** | **0.16** | **0.10** |

SFT/DPO 가 x_ai(탐지기가 확실히 AI로 잡던 것)를 인간 영역으로 이동시킨다. 붕괴 없이(신문 SFT 붕괴율 0%).

---

## 파이프라인 (Stage 1–3, 도메인 무관)

```
[Stage 1] 데이터 구축
  1a 인간 원문 추출  ── 도메인별(유일한 도메인 의존 단계). 본문 800~1200자(ko-BART 512토큰 안)
     │  human_pool.jsonl
  1b LLM 재서술(x_ai) ── P3b 통일 프롬프트 + Qwen3-8B(temp 0.8, no-think)
     │  gen.jsonl                build_domain_prompts.py → generate.py
  1c 게이트          ── 도메인 탐지기 D 를 detector-split 으로 1회 학습·동결
     │  dpair.jsonl              → d_human<τ AND d_ai≥τ 통과쌍만 (news_gate_frozen.py)
  1d 조립(선택)      ── 게이트를 여러 번(코어·확장·top-up) 합칠 때: D학습 제외·기사 dedup·분할 (build_clean_dataset.py)
     │  clean 데이터셋           → 누수 0 을 assert 로 검증
[Stage 2] Style SFT  ── StyleBART(ko-BART, concat fusion + EOS), 이중경로 λ0.5 (stage2_sft.py)
     │  models/{dom}_sft
[Stage 3] DPO        ── DPOP λ5, chosen=x_human / rejected=SFT hard-neg(D>τ) / ref=SFT
        models/{dom}_dpo           stage3_build_dpo.py → stage3_dpo.py
```

원논문 MASH 처럼 **하나의 재작성 절차를 전 도메인에 균일 적용**한다(도메인별 프롬프트 분기 없음).
새 도메인은 `human_pool.jsonl` 만 준비하면 `run_domain_pipeline.sh <domain> <gpu>` 로 1b→3 이 돈다.
정본 규격 전문: **[notes/51-Stage1-3-파이프라인-정본](notes/51-Stage1-3-파이프라인-정본.md)**.

### 정본 학습 설정 (EOS SFT + DPOP)

| 단계 | 스크립트 | 정본 설정 |
| --- | --- | --- |
| SFT | `stage2_sft.py` | **concat→proj fusion**(2d→d, 항등 초기화) + **라벨에 EOS 추가**. `--epochs 3 --bs 8 --lr 2e-5 --lam 0.5 --maxlen 512` (전부 기본값) |
| DPO | `stage3_dpo.py` | **`--dpop --dpop-lambda 5.0 --length-norm --beta 2.0 --epochs 1 --bs 2 --accum 8 --lr 5e-6`** (유효배치 16) |

- concat fusion 은 이전 additive(`cr+fusion(cr+style)`)의 content 오염·붕괴(SFT 30%/DPO 53%)를 해결한 것이고,
  EOS 는 길이 폭주·문장 반복을 막는다. 둘 다 현행 코드의 **기본 동작**이다.
- ⚠️ **`stage3_dpo.py` 의 기본값은 vanilla DPO(β0.1, epochs 5, DPOP off)** 다. 그대로 돌리면 SFT 에서 과이탈해
  붕괴한다(reward hacking). **정본은 위 DPOP 플래그를 반드시 명시**해야 하며, `run_domain_pipeline.sh` 가 그렇게 호출한다.
- 정본 계보(신문): `news_sft_eos20k`(concat+EOS) → DPOP λ5 → `news_dpo_eos_dpop_l5`.
- 상세·재현 커맨드: [notes/44](notes/44-Stage2-EOS-Stage3-DPO-DPOP.md).

### 탐지기 D — 학습과 평가를 겹치지 않게

- **도메인별 klue/roberta 파인튜닝, 1회 학습 후 동결.** 게이트 · DPO 보상 · 최종 평가에 같은 D 하나를 쓴다.
- **D 가 학습한 기사(detector-split)는 D 가 채점하는 pair 풀에서 제외한다** — 안 그러면 D 가 자기 학습 문서를
  채점해 게이트를 통과시키는 누수가 된다. `news_gate_frozen.py` 는 detector-split 을 seed-42 로 고정 분리하고
  그 문서는 스코어링·게이트 루프에서 아예 빠진다. 확장(`--frozen-only`) 경로에도 제외 가드가 있다.
- **누수 감사**: `audit_overlap.py` 로 `clean20k ∩ D학습 = 0`, `train ∩ test = 0` 을 검증한다(D held-out AUROC 0.9913).
  ⚠️ 과거 확장 경로에서 가드가 없어 722편이 누수된 사고가 있었고(→ `news_dpair_final.jsonl`, **폐기·사용금지**),
  이를 제거해 `clean20k` 를 만든다 — 이 **조립·정제·분할은 `build_clean_dataset.py` 가 코드로 못박아 재현 가능**하다
  (제외 id 를 빼고 기사 단위 dedup·분할, 누수 0 을 assert). 오염본 20,588행 → 누수 722편 전량 제거 검증 완료.
- **더 강한 D2**(4,500편·3생성기 qwen/exaone/kanana)로 교체 중. D2 추출은 `mash_extract_pool.py --exclude` 로
  D2 split 과 clean20k 를 애초에 배제해 누수를 원천 차단한다.

### 평가·시각화

| 축 | 스크립트 | 내용 |
| --- | --- | --- |
| 회피(타깃/전이) | `roberta_gate_scores.py`*, `scrn_train.py`+`scrn_score.py`, `detect_binoculars.py`, `detect_fastdetectgpt.py`, `detect_llm_judge.py` | roberta-D(타깃) · SCRN(도메인별 독립, koelectra) · Binoculars/FastDetect(zero-shot 전이) · LLM 판단(Russell) |
| 의미 보존 | `metrics_quality.py` | BERTScore-F1(후보 vs x_ai) |
| 붕괴·유창성 | `eval_collapse.py` | clean20k test 에서 붕괴율·rep-n·D·사람판정·cos |
| 시각화 | `analyze_domain.py` | roberta-D CLS 특징 t-SNE + D/SCRN P(AI) |
| 검수 | `make_dpo_review.py`, `make_stage2_review.py` | SFT/DPO 4단 비교 HTML |

SCRN 은 **도메인별로 D(D2)와 같은 split 으로 학습**해야 한다(같은 데이터·다른 아키텍처 = "D 하나에 과적합 아님"
증명). 학습된 분류기를 out-of-domain 에 쓰면 전이가 아니라 오작동이므로, **미지 탐지기 전이**는 학습이 없는
zero-shot(Binoculars/FastDetect)이 담당한다. (`*roberta_gate_scores.py` 는 초록 트랙의 구 5-fold 방식이며 신문
트랙은 frozen-D 로 대체됨 — [notes/50](notes/50-파이프라인-현황.md).)

> **회피 수치는 ②의미보존·③붕괴를 통과해야 유효하다.** 붕괴·의미파괴로 낮춘 탐지 점수는 "가짜 성공"이다.
> vanilla DPO 의 높은 회피율은 반복 루프 붕괴가 탐지기를 헷갈린 허수였고, 그래서 DPOP 로 교체했다.

---

## 저장소 구성

```
scripts/   Stage 1–3 파이프라인 + 평가/시각화 + (초록 트랙) 수집·검사
notes/     연구 노트 (진입점: notes/README.md). 정본: 44(SFT/DPO)·45(도메인확장)·51(파이프라인)·60(실험설계)
mash/      초록(KCI) 트랙 — 데이터셋 명세 · 파이프라인 규격 · 확정 프롬프트
archive/   구버전 스냅샷 · 회의록 (읽기 전용)
```

- 상태·다음 할 일: [progress.md](progress.md)
- Stage 1–3 정본 규격: [notes/51](notes/51-Stage1-3-파이프라인-정본.md)

핵심 스크립트:

```
scripts/
  build_domain_prompts.py   1b · P3b 통일 프롬프트 (전 도메인 공용)
  generate.py               1b · llama-server 병렬 생성 (--resume)
  news_gate_frozen.py        1c · D 동결 학습 + 게이트(누수 제외 가드 포함)
  build_clean_dataset.py    1d · 게이트 결과 합쳐 누수 없는 데이터셋 조립(제외·dedup·분할·검증)
  stage2_sft.py             2  · StyleBART(concat+EOS) SFT
  stage3_build_dpo.py       3  · hard-neg 채굴
  stage3_dpo.py             3  · DPOP 학습
  run_domain_pipeline.sh    1b→3 도메인 오케스트레이터
  analyze_domain.py         평가 · t-SNE + D + SCRN
  scrn_train.py/scrn_score.py, detect_*.py, metrics_quality.py, eval_collapse.py, audit_overlap.py
```

---

## 재현

### 0. 준비

```bash
git clone https://github.com/sybyrr/AIKU-LLM-Text-Humanization.git
cd AIKU-LLM-Text-Humanization
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**데이터는 이 저장소에 없다**(위 경고 참조). 참여자는 Google Drive 묶음을 받아 `dataset/`·`models/` 에 배치한다.

### 1. 새 도메인에 파이프라인 적용

`dataset/{dom}_track/human_pool.jsonl`(본문 800~1200자, pre-ChatGPT) 을 준비하고 qwen3-8b llama-server 를 띄운 뒤:

```bash
bash scripts/run_domain_pipeline.sh <domain> <gpu>
# 1b 프롬프트→생성 → 1c 게이트(D 동결) → 2 SFT(concat+EOS) → 3 DPOP λ5
```

- 도메인이 바뀌어도 프롬프트·게이트·SFT·DPO 는 그대로다. 바뀌는 것은 **1a 추출기**(도메인마다 새로 작성)뿐이다.
- 초록(KCI) 트랙의 1a 수집·검사는 `collect_kci.py`·`stage0_*` 계열을 쓴다(상세: 아래 mash 규격).

### 2. 평가

```bash
.venv/bin/python scripts/eval_collapse.py --ckpt models/{dom}_dpo/dpo_bart.pt --split test --pairs <clean20k>
.venv/bin/python scripts/analyze_domain.py --domain {dom} --sft <sft.pt> --dpo <dpo.pt> \
    --roberta models/{dom}_roberta_D --pairs <dpair> --out review/{dom}_tsne.png
```

평가는 **clean20k 의 test split(held-out)** 에서만 낸다. ⚠️ 구 `transfer_*.jsonl`(2026-08-17)은 v2-test 기반이라
clean20k train 과 116/150 누수 → **폐기**. 인용 금지.

---

## 공개 범위 · 라이선스

2026-08 결정: 저장소는 **공개**, 데이터는 **Drive 로 참여자에게만**. `.gitignore` 가 경계를 강제한다.

| | 올라가는 것 | 이유 |
| --- | --- | --- |
| GitHub(공개) | `scripts/` · `notes/` · `mash/*.md` · `archive/`(문서) · 루트 문서 | 코드·연구 기록. **코퍼스 텍스트 0건** |
| Drive(비공개) | 인간 풀·생성 pair·프롬프트·모델·docx·PDF·검수 HTML | 저작권·약관상 재배포 불가 |

**커밋 금지**(검사 확인): `.API_KEY`·`.kli_api_key`, `dataset/`·`models/`·`review/`·`logs/` 전량,
모든 `*.jsonl`(프롬프트 포함), `*.docx`/`*.pdf`, `archive/aihub_essay_2024_raw/`(AI Hub 약관), 팀원 실명 자료(동의 전).

---

## 열린 것 / 다음

- **청원·위키 재생성** — 두 도메인은 과거 `rewrite_ko`(문장 순서 유지 = 복사 유발 P2 회귀본)로 생성돼 신문(P3b)과
  불일치. `build_domain_prompts.py`(P3b 통일)로 재생성 + D2 재게이트.
- ~~clean20k 조립 코드화~~ ✅ **`build_clean_dataset.py` 로 코드화** — 제외·dedup·분할·누수 0 assert. 사후 수동 단계 해소.
- **petition/wiki D2·SCRN 구축** — 현재 파일럿 탐지기(750편·단일 생성기)라 회피 수치가 뭉툭. 공정 비교엔 확대 필요.
- **에세이(545) 신청·추출기**, **Stage 4(추론시 정제)** 미탐색, **평가 스크립트 통합**(test 전편 × 전탐지기 + t-SNE).
