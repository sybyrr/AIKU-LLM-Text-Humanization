# 51 · Stage 1–3 파이프라인 정본 (도메인 무관)

> 2026-08-26 세션 정리. Stage 1(데이터 구축) → 2(SFT) → 3(DPO)를 **도메인 무관 정본**으로 고정한다.
> 상위: [50-파이프라인-현황](50-파이프라인-현황.md)(신문 08-16, 일부 대체) · [60-실험설계](60-실험설계.md)
> 도메인확장 실측: [45-도메인확장-청원-위키](45-도메인확장-청원-위키.md)
>
> 원논문 MASH = **inverse data construction**: 인간글을 LLM으로 재작성해 x_ai 를 만든 뒤 탐지기로 게이트.
> 도메인별 프롬프트 분기 없음(하나의 재작성 절차를 전 도메인 균일 적용). 우리 P3b 가 그 한국어판이다.

---

## 0. 한 문장

인간 원문 → LLM 재서술(x_ai) → 도메인 탐지기 게이트로 `(x_human, x_ai)` 쌍 확보 → StyleBART SFT 로
x_ai→x_human 문체 전환 → DPOP 로 회피 강화. **전 도메인 동일 절차·동일 프롬프트.**

## 1. 전체 흐름

```
[Stage 1] 데이터 구축
  1a 인간 원문 추출  ── 도메인별(유일한 도메인 의존 단계). 800~1200자 필터
     │  human_pool.jsonl
  1b LLM 재서술(x_ai) ── ★P3b 프롬프트 통일. Qwen3-8B temp0.8
     │  gen.jsonl
  1c 게이트          ── 도메인 D 동결 학습 → d_human<τ AND d_ai≥τ 통과쌍만
     │  dpair.jsonl (train/dev/test, 기사 단위)
[Stage 2] Style SFT  ── StyleBART(concat fusion + eos), 이중경로 λ0.5
     │  models/{dom}_sft
[Stage 3] DPO        ── DPOP λ5, chosen=x_human / rejected=SFT hard-neg / ref=SFT
        models/{dom}_dpo
```

탐지기 D 는 전 과정 동일(게이트 = DPO 보상 = 평가). **타깃은 D2 로 이관 중**(§6).

---

## 2. Stage 1 — 데이터 구축

### 1a. 인간 원문 추출 (도메인별 — 유일한 도메인 의존 단계)

- 공통 필터: 본문 **800~1200자** — ko-BART 인코더 512토큰 안(실측 최대 561토큰). 그 위는 잘려서 붕괴.
  책·판결문 등 장문 소스는 이 창으로 **청킹**해서 넣는다(문단 경계, 짧으면 인접 문단 묶음).
- 오염 컷: 본문 작성 시점 **2022-11-30(ChatGPT 공개) 이전**만. 소스별로 연도/날짜 필드나 본문 단서로 확인.
- 추출기는 도메인마다 다름(신문 ZIP 파서 / HF 로더 / AI Hub 파서 …). 아래 1b~3 은 전부 공용.
- 출력 스키마: `human_pool.jsonl` = `{doc_id, text|body, n_char, title, …}`

### 1b. LLM 재서술 → x_ai  (★프롬프트 통일 = 이번 세션 핵심 결정)

- **정본 프롬프트 = P3b 하나("글" 버전).** 도메인 단어 없이 전 도메인 공용. → `scripts/build_domain_prompts.py`
  (system/user 전문은 그 파일 상단. 초록→신문 이식 때 바꾸던 2군데를 없애고 도메인 무관으로 고정.)
- ⚠️ **폐기: `scripts/prompts/mash/rewrite_ko.user.txt`** — "문장 순서 유지"가 들어간 **P2 회귀본**이다.
  P3b 가 일부러 제거한 문장 단위 대응(복사 유발 → 많이 베낀 글은 게이트에서 인간 판정, r=-0.99)을 도로 넣었다.
  **청원·위키가 이걸 써서 신문(P3b)과 불일치** → 두 도메인 **재생성 필요**(§6).
- 생성: `generate.py` + **Qwen3-8B**(GGUF Q4_K_M, llama-server). `--temp 0.8 --seed 42 --no-think --token-ratio 0.8`.
  temp 0.8 근거: 그리디(0.0)가 원문을 +3.0%p 더 베끼고, 많이 베낀 글은 게이트 탈락.
- D2 타깃이면 **3생성기 라운드로빈**(qwen / exaone / kanana)로 생성해 생성기 편향 제거.

### 1c. 게이트 → dpair (도메인별 frozen 탐지기)

- `news_gate_frozen.py`: detector-split 으로 D 를 **1회 학습·동결** → 게이트: **`d_human < τ` AND `d_ai ≥ τ`**(τ=0.5)
  통과 쌍만 학습 데이터. D 학습 기사는 pair 풀에서 제외 → 누수 0.
- **타깃 = D2**(klue-roberta, `--det-n 4500`, 3생성기). 현 D(750·2생성기)에서 이관.
- 출력: `dpair.jsonl` — **기사 단위** train/dev/test 분할(train∩test=0).

### 1d. 조립 (선택 — 게이트를 여러 번 합칠 때)

게이트가 코어·확장·top-up 으로 여러 번 돌면 결과가 흩어지고, 확장분(`--frozen-only`)에 D 학습 기사가
섞여 들어갈 수 있다(과거 `news_dpair_final` 의 **722 누수**). `build_clean_dataset.py` 가 **합치기 → D학습
제외 → 기사 단위 dedup(2-arm 중복 폐기) → 기사 해시 분할**을 코드로 못박고 `누수 0·중복 0·split 분리`를
assert 로 검증한다(오염본 20,588→722 제거 검증됨). 단일 실행이면 1c 의 dpair 를 그대로 쓰고, 병합이 필요할 때만 탄다.

---

## 3. Stage 2 — Style SFT (StyleBART)

- **ko-BART(124M) + learnable ASR/HSR 스타일 벡터 + concat→proj fusion.**
  fusion = `Linear(2d→d)` on `concat[content, style]`, zero-init 로 초기 항등.
  ⚠️ 이전 **additive**(`cr + fusion(cr+style)`)는 content 를 오염시켜 붕괴(SFT 30%/DPO 53%). **concat 이 정본**(현행 `stage2_sft.py` 기본).
- **EOS**: 라벨 끝에 `eos` 직접 추가(ko-BART 토크나이저는 자동으로 안 붙임) → 길이 폭주·문장 반복 방지. 현행 기본.
- 이중경로: 경로1 `CR+ASR → x_ai 복원`(L_recon, 의미 보존), 경로2 `CR+HSR → x_human`(L_trans, 문체 이식).
  `L = λ·L_recon + (1-λ)·L_trans`, **λ=0.5**.
- 실행: `stage2_sft.py --pairs dpair --out-dir models/{dom}_sft --epochs 3 --bs 8 --lam 0.5 --split train`
- **정본 산출(news)**: `models/news_sft_eos20k` (concat + eos, clean20k).

---

## 4. Stage 3 — DPO (DPOP)

- chosen = x_human, rejected = SFT 가 HSR 로 생성한 것 중 **D>τ**(hard-neg), ref = SFT(고정), 보상 = 게이트와 같은 D.
- 하드넷 채굴: `stage3_build_dpo.py --sft models/{dom}_sft/style_bart.pt --pairs dpair --roberta models/{dom}_roberta_D --tau 0.5 --n-sample 4 --split train --out dpo_pairs.jsonl`
- 학습: `stage3_dpo.py --dpo-pairs dpo_pairs --sft …/style_bart.pt --out-dir models/{dom}_dpo`
  **`--dpop --dpop-lambda 5.0 --length-norm --beta 2.0`** `--epochs 1 --bs 2 --accum 8 --lr 5e-6`
- **DPOP λ5 + length-norm + β2.0 이 정본.** 표준 DPO(β0.1)는 SFT 에서 과이탈해 붕괴("출렁다리 출렁다리…").
  DPOP 은 chosen 이 ref 밑으로 떨어질 때만 hinge 페널티 → 붕괴 없이 소폭 회피 개선(변화폭 = hard-neg 수 비례).
- **정본 산출(news)**: `models/news_dpo_eos_dpop_l5`.

---

## 5. 정본 하이퍼·모델 (news 기준, 도메인은 {dom} 치환)

| 단계 | 스크립트 | 핵심 인자 | 산출물 |
| --- | --- | --- | --- |
| 1b 프롬프트 | `build_domain_prompts.py` | P3b 통일("글") | `prompts.jsonl` |
| 1b 생성 | `generate.py` | qwen3-8b · temp0.8 · no-think · ratio0.8 | `gen.jsonl` |
| 1c 게이트 | `news_gate_frozen.py` | `--det-n 4500(D2)` · τ0.5 · 3 arm | `dpair.jsonl` + `{dom}_roberta_D2` |
| 2 SFT | `stage2_sft.py` | concat+eos · epochs3 · bs8 · λ0.5 | `models/{dom}_sft`(=news `_eos20k`) |
| 3 build | `stage3_build_dpo.py` | τ0.5 · n-sample4 | `dpo_pairs.jsonl` |
| 3 DPO | `stage3_dpo.py` | dpop λ5 · length-norm · β2.0 · 1ep · 유효배치16 | `models/{dom}_dpo`(=news `_eos_dpop_l5`) |

---

## 6. 도메인별 상태

| 도메인 | 인간 소스 | 규모 | 상태 |
| --- | --- | --- | --- |
| **초록(메인)** | KCI 논문초록 | 9.4k(+API 확장가능) | D 원조, flip 77% 검증 |
| **신문** | 신문말뭉치 clean20k | pair 20,343 | Stage 1–3 완료 |
| 청원 | HF `dev7halo/bluehouse-national-petition` | pair 1,425 | ⚠️ rewrite_ko 로 생성 → **P3b 재생성 필요** |
| 위키 | HF `wikimedia/wikipedia 20231101.ko` | pair 1,197 | ⚠️ 동상 + 리스트 문체 붕괴 |
| **에세이** | AI Hub `dataSetSn=545`(2021, KatFish 출처) | 50,413편 | 신규 후보(신청). 논술형+길이하한 필터 |
| **판결문** | 법제처/HF `joonhok-exo-ai/…precedents` | 8.5만+ | 신규 후보(규모 무제한, 클린) |
| ~~QA~~ | 법률상담사례 등 | ≤5천 | 보류 — ≥20k 클린 장문 소스 없음(소규모 전이만 가능) |

## 7. 이번 세션 확정 결정

1. **프롬프트 P3b 하나로 통일("글")** — 원논문도 도메인 균일 적용. `rewrite_ko`(순서유지 P2회귀) 폐기 → 청원·위키 재생성.
2. **SCRN = 도메인별, D2 와 같은 split 으로 학습** — 독립 2번째 탐지기(다른 arch=koelectra, "D 과적합 아님" 증명).
   전이/unseen 축은 **Binoculars·FastDetect(zero-shot)** 담당(학습 없어 도메인 무관). 학습된 분류기를 out-of-domain 에 쓰면 전이가 아니라 오작동(위키 인간글 0.58 오판).
3. **SFT = concat+eos, DPO = DPOP λ5** 를 정본 체크포인트로.
4. **평가 = clean20k test 전편(2044)** × [D2 · SCRN · Binoculars · FastDetect · LLM판단(32B)] + **유사도 4×4 행렬**(문서별 cos: cos(x_ai, SFT/DPO)=내용보존, cos(human, SFT/DPO)=근접도) + **t-SNE 전체점**. 표본 없음(비싼 32B 판단만 시간 소요). 지표축: 회피=TPR@FPR5% / 의미보존=cos·BERTScore / 붕괴=rep-n·PPL(회피의 게이트).

## 8. 열린 것 / 다음

- 청원·위키 **P3b 재생성 + D2 재게이트**(rewrite_ko 오염 제거).
- petition/wiki **D2·SCRN 구축**(현재 news 만 있음).
- 에세이(545) **신청·전용 추출기**(논술형·길이하한·초등 제외).
- **Stage 4**(추론시 정제) 미탐색.
- **평가 스크립트 1개로 통합** — 4단계 생성·채점을 test 전편으로 한 번 돌려 표·t-SNE 양쪽에 재사용.

## 9. 스크립트 맵

```
scripts/
  build_domain_prompts.py   1b · P3b 통일 프롬프트 조립 (rewrite_ko 대체)  [신규]
  run_domain_pipeline.sh    1b→1c→2→3 도메인 오케스트레이터            [신규]
  generate.py               1b · llama-server 병렬 생성
  news_gate_frozen.py        1c · D 동결 학습 + 게이트(--frozen-only 제외 가드)
  build_clean_dataset.py    1d · 게이트 결과 조립(제외·dedup·분할·누수 0 검증)  [신규]
  stage2_sft.py             2  · StyleBART(concat+eos) SFT
  stage3_build_dpo.py       3  · hard-neg 채굴
  stage3_dpo.py             3  · DPOP 학습
  analyze_domain.py         평가 · t-SNE + D + SCRN
  prompts/mash/rewrite_ko.user.txt   ⚠️ 폐기(P2 회귀본) — build_domain_prompts.py 로 대체
```
