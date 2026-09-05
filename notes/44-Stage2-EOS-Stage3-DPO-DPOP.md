# 44 · Stage2(EOS 수정) & Stage3(DPO → DPOP) 작업 기록

> 목적: ko-BART humanizer의 Stage2 SFT / Stage3 DPO 재현 중 발견한 두 버그(EOS 미학습, DPO degeneration)와
> 그 해결(DPOP)을 **재현·검증 가능하게** 기록. 애매함 없이 수치·코드·커맨드·경로까지 명시.
> 작성 시점 기준 모든 Stage3 실험은 **1,561쌍(n_sample=1) · 옛 detector D** 에서 수행. n4(4,605)는 미학습.

---

## 0. 결론 (TL;DR)

- **Stage2 버그**: ko-BART 토크나이저가 EOS(`</s>`)를 라벨에 안 붙임 → 모델이 멈추는 법을 못 배움 →
  추론 시 `max_length`까지 생성하며 반복/길이폭주. **수정: 라벨 끝에 eos 수동 추가.** → SFT 정상화.
- **Stage3 버그**: vanilla DPO가 **reward hacking(붕괴)** 로 detector를 속임. 원인 = **chosen unlearning**
  (chosen·rejected가 x_ai 패러프레이즈라 토큰 공유 → 차이기반 손실이 chosen 확률까지 끌어내림).
- **해결 = DPOP** (chosen이 ref 밑으로 갈 때만 hinge 페널티). β·epoch·lr·SimPO·길이정규화·RPO 다 시도했고
  **DPOP가 저울(λ)로 가장 깔끔하게 제어**됨.
- **채택: DPOP.**
  - `DPOP λ5` = **SFT보다 깨끗하게 나은 유일한 지점** (D 0.252→0.179, 사람판정 77→85%, gemma 붕괴 12%≈SFT 10%).
  - **강한 회피(D≈0.05, 97%)는 방법 불문 붕괴 필연** (약한 옛-D를 붕괴로 속이는 것 말고 길이 없음) → 원하면 Stage4로 사후청소.

---

## 1. 파이프라인 컨텍스트

- Stage1: 코퍼스(x_ai = qwen3-8b 75% / exaone-3.5-7.8b 25% 생성 뉴스, x_human = 실제 사람 뉴스).
- Stage2 SFT: ko-BART(`gogamza/kobart-base-v2`) + style-injection(ASR/HSR + concat fusion). x_ai→x_human 스타일 이식.
- Stage3 DPO: SFT를 detector 회피 방향으로 정렬. prompt=x_ai, chosen=x_human, rejected=SFT의 hard-neg.
- **Detector D 라인 통일**: `models/news_roberta_D` (옛 D, 750개 학습분). 데이터게이팅·DPO·평가 모두 이 D 사용.
  (D2 `models/news_roberta_D2`는 별개; §7 참고.)

---

## 2. Stage2 — EOS 버그와 수정

### 증상
- 패러프레이즈 출력에 문장 단위 반복 / 중간 짤림. beam search로도 안 잡힘.

### 원인 (실측)
- `gogamza/kobart-base-v2` 토크나이저는 `add_special_tokens` 기본값에도 **bos/eos를 안 붙임**.
  (`tok("문장").input_ids` 앞뒤에 `<s>`(id 1)·`</s>`(id 1) 없음. special: bos=1, eos=1, pad=3, decoder_start=1.)
- `stage2_sft.py`가 라벨을 `tok(texts).input_ids` 그대로 사용 → **학습 타깃에 eos가 한 번도 없음**
  → 모델이 "여기서 멈춰라(eos 방출)"를 못 배움 → 추론 시 `max_length`까지 생성 → 반복으로 공간 채움.
- 실측: 입력 855자 → SFT 출력 2092자(2.5배), no_repeat 제거 시 문장중복 18%.

### 수정 (`scripts/stage2_sft.py` `collate()`의 `labels()`)
- 각 타깃 끝에 `tok.eos_token_id` 수동 추가 후 pad(-100). **인코더 입력엔 안 붙임**(SFT가 인코더엔 eos 없이 학습되므로 일관성 유지).
- 참고: 영어 `facebook/bart-base`(원논문)는 토크나이저가 eos를 자동으로 붙여서 이 버그가 없었음 → 한국어 포팅 시 새로 생긴 구멍.

### 결과 (재학습, `models/news_sft_eos20k`)
- 학습: `news_dpair_clean20k.jsonl` **train split 16,266쌍**, 3 epochs, bs 8, lam 0.5.
- held-out test(plain beam4): 붕괴 0%, 길이 178어절(입력 ~194), **D 0.252(사람판정 77%)**, cos 0.984, gemma 붕괴 10%. → 정상.

---

## 3. Stage3 — DPO degeneration 진단

### 증상
- vanilla DPO(β0.1, 5ep 등) 학습 후 출력이 붕괴(고유명사·숫자 오염, 문장 재진술, 반복). detector D는 0.007까지 내려가나
  이는 **가짜 회피**(붕괴 텍스트가 D 학습분포 밖이라 human으로 오분류). 내용보존(cos) 폭락.

### 진단 지표 (계측 추가)
- `stage3_dpo.py` 학습 로그에 **rew_c / rew_r** 추가.
  - `rew_c = β·(logπ_policy(chosen) − logπ_ref(chosen))` = chosen이 SFT 대비 얼마나 더/덜 나오나. **음수 = chosen unlearning.**
  - 건강한 DPO = rew_c≈0 유지 + rew_r만 하락. 우리 vanilla = **rew_c·rew_r 둘 다 곤두박질**.
- 손실 자체는 정상(검증: policy==ref일 때 loss=log2 정확, seq_logprob=HF loss 일치). **버그 아니라 vanilla DPO의 구조적 문제.**

### 근본 원인 (Meta Llama-3 보고와 동일)
- chosen(x_human)·rejected(SFT 출력)가 **둘 다 x_ai 패러프레이즈 → 토큰 다수 공유**.
- 차이기반 손실은 (chosen−rejected) margin만 키우면 되므로, **공유 토큰(=chosen 포함)을 내리는 게 최적해** → chosen unlearning → 붕괴.
- 추가로 sum-기반 로그우도는 chosen(x_human)이 rejected보다 길어(중앙 421 vs 360 토큰) **margin이 초기 −240으로 폭발**(길이 bias).

---

## 4. 방법 비교 (전부 1,561쌍 · plain beam4 · gemma 확정)

| 설정 | 손실 변경 | D | 사람판정 | rule붕괴 | **gemma붕괴** | cos | 판정 |
|---|---|---|---|---|---|---|---|
| human (참고) | — | 0.043 | 100% | — | 2% | — | — |
| **SFT (기준)** | — | 0.252 | 77% | 0% | **10%** | 0.984 | 깨끗 |
| DPO β0.1/5ep | vanilla | 0.007 | 100% | 77% | 100% | 0.765 | 붕괴 |
| DPO β0.5/1ep | β↑ | 0.033 | 98% | 37% | 82% | 0.906 | 붕괴 |
| DPO lr1e-6 | lr↓ | 0.033 | 98% | 30% | — | 0.923 | 붕괴 |
| 길이정규화 DPO β2 | length-norm | 0.005(dev) | — | — | — | — | 붕괴(겹침0.00) |
| SimPO β2 γ1 | ref-free+γ | — | — | — | — | — | 붕괴(겹침0.00) |
| RPO α0.5 (+ln β2) | +chosen NLL | 0.039 | 97% | 8% | **52%** | 0.968 | 부분개선·붕괴 |
| **DPOP λ1 (+ln β2)** | hinge(rc>pc) | 0.052 | 97% | 7% | **57%** | 0.968 | 강회피·붕괴 |
| **DPOP λ5 (+ln β2)** | hinge(rc>pc) | **0.179** | **85%** | **2%** | **12%** | 0.982 | ✅ **깨끗한 개선** |

핵심:
- **하이퍼파라미터(β/epoch/lr)만으론 못 고침** — 전부 붕괴.
- **SimPO·길이정규화 = 차이기반이라 chosen unlearning 못 막음** — 붕괴(원문겹침 0.00).
- **RPO / DPOP = chosen을 붙잡는 항 추가 → rew_c 안정.** 근데 강회피 지점(D≈0.05)에선 RPO(52%)·DPOP λ1(57%) 둘 다 여전히 붕괴.
- **DPOP λ(앵커 강도)가 "회피↔품질" 저울**: λ5=깨끗+약회피 / λ1=강회피+붕괴. **λ5가 SFT보다 깨끗하게 나은 유일 지점.**

**결론 명제**: *약한 옛-D에서는 "SFT보다 깨끗하면서 강하게 회피"하는 지점이 존재하지 않는다.*
D를 ~0.05까지 내리려면 반드시 (부분)붕괴로 detector를 속여야 함(방법 불문). 깨끗하게 얻을 수 있는 개선폭은 D 0.252→0.179(77→85%)까지.

---

## 5. 채택: DPOP

- **방법 = DPOP** (Smaug / DPO-Positive). vanilla DPO 손실의 margin 안에 **chosen이 ref 밑(rc>pc)으로 갈 때만** 발동하는 한쪽 hinge 페널티.
- **깨끗 우선 → `DPOP λ5`** (D 0.179 / 85% / gemma 12%). **강회피 필요 → `DPOP λ1`(또는 RPO) + Stage4 사후청소.**
- 표준성 근거: Meta Llama-3(chosen NLL α=0.2), RPO(NeurIPS 2024), TRL `rpo_alpha`, Smaug DPOP — chosen 보존항은 널리 쓰이는 표준 처방.

---

## 6. 정확한 재현 정보

### 6.1 코드 변경 (`scripts/`)
- `stage2_sft.py` · `collate()`/`labels()`: 타깃 끝에 `tok.eos_token_id` 추가, pad=-100.
- `stage3_dpo.py`:
  - `seq_logprob(..., norm=False)`: `norm=True`면 로그우도를 토큰당 평균(길이정규화).
  - 인자 추가: `--length-norm`, `--simpo`/`--gamma`, `--rpo-alpha`, `--dpop`/`--dpop-lambda`.
  - 학습 로그에 `rew_c`/`rew_r` 출력.
  - **DPOP 손실**: `h = (pc-rc)-(pr-rr); if dpop: h = h - λ·relu(rc - pc); logits = β·h`.
  - RPO: `loss += α·(−ℓθ(chosen))` (chosen 토큰당 NLL).
- `eval_collapse.py` · `DECS`: **plain beam4** 로 변경 `{"plain": dict(num_beams=4, max_length=1024)}`.
  (no_repeat/rep_pen은 EOS수정 전 크러치라 붕괴를 **가림** → 평가에 쓰지 말 것.)

### 6.2 커맨드 (환경: `PY=/workspace/.venv/bin/python`, `cd scripts`, `LD_LIBRARY_PATH` export 필요)

Stage2 SFT:
```
CUDA_VISIBLE_DEVICES=0 $PY stage2_sft.py \
  --pairs dataset/news_track/news_dpair_clean20k.jsonl \
  --out-dir models/news_sft_eos20k --epochs 3 --bs 8 --lam 0.5 --split train
```

Stage3 DPO 데이터(hard-neg 채굴, n_sample=1 = 논문 방식; --batch 로 배치 생성):
```
$PY stage3_build_dpo.py --sft models/news_sft_eos20k/style_bart.pt \
  --pairs <train shard> --roberta models/news_roberta_D \
  --tau 0.5 --n-sample 1 --batch 48 --out <shard out>
# 4-GPU 샤딩 후 병합 → dataset/news_track/dpo_pairs_eos.jsonl (1,561쌍)
# n_sample=4 버전 → dpo_pairs_eos_n4.jsonl (4,605쌍, 아직 미학습)
```

Stage3 DPOP (**채택 세팅, λ5**):
```
CUDA_VISIBLE_DEVICES=0 $PY stage3_dpo.py \
  --dpo-pairs dataset/news_track/dpo_pairs_eos.jsonl \
  --sft models/news_sft_eos20k/style_bart.pt \
  --out-dir models/news_dpo_eos_dpop_l5 \
  --beta 2.0 --dpop --dpop-lambda 5.0 --length-norm \
  --epochs 1 --bs 2 --accum 8 --lr 5e-6 \
  --eval-pairs dataset/news_track/news_dpair_clean20k.jsonl --roberta models/news_roberta_D
# 강회피판: --dpop-lambda 1.0 (out-dir news_dpo_eos_dpop_l1)
```

### 6.3 하이퍼파라미터 (DPOP 채택)
| 항목 | 값 | 비고 |
|---|---|---|
| loss | DPO + DPOP hinge | `h -= λ·relu(rc−pc)` |
| length_norm | True | 로그우도 토큰당 평균 (margin 폭발 방지) |
| β | 2.0 | 길이정규화 스케일이라 큼(vanilla는 0.1) |
| **λ (dpop-lambda)** | **5.0(깨끗) / 1.0(강회피)** | 회피↔품질 저울 |
| epochs | 1 | DPO는 과최적화 쉬워 1ep |
| lr | 5e-6 | |
| bs × accum | 2 × 8 (유효 16) | |
| ref | SFT `news_sft_eos20k`(동결) | |

### 6.4 산출 모델
- `models/news_sft_eos20k/style_bart.pt` — SFT(EOS 수정)
- `models/news_dpo_eos_dpop_l5/dpo_bart.pt` — **DPOP λ5 (채택·깨끗)**
- `models/news_dpo_eos_dpop_l1/dpo_bart.pt` — DPOP λ1 (강회피·붕괴, Stage4 전제)

---

## 7. 평가 방법 (정확히 — 재검증용)

- **디코딩 = plain beam4** (`num_beams=4, max_length=1024`, 반복방지 없음). = 논문 세팅이자 정직한(안 가린) 평가.
  ※ 내장 dev 검증(`stage3_dpo.py` 끝)은 no_repeat 디코딩이라 붕괴를 **가림** — dev의 원문겹침/ D는 참고만.
- **rule 붕괴율** = 한 어절 30회 초과 반복 문서 비율. **거침, 미묘한 붕괴 놓침**(rule 8% vs gemma 52% 사례) → 단독으로 믿지 말 것.
- **gemma 붕괴율 = 신뢰 지표.** `judge_collapse.py` + `Gemma-3-27b`(중립 judge). reference-free 이진(정상/붕괴) 판정.
  - judge 선정 근거: 생성자(qwen3-8b/exaone)와 **다른 패밀리**여야 self-detection 회피. gemma는 3-judge(gemma/kanana/midm) 캘리브레이션에서 x_ai 0%·human 2%·명백붕괴 88~98%로 **가장 정확**(midm은 신뢰 낮음).
- **D** = `news_roberta_D` P(AI), τ=0.5. 낮을수록 회피. **주의: D는 붕괴탐지기가 아님** — 붕괴 텍스트도 human으로 봄(D2도 동일).
- **cos** = ko-sroberta 문장 임베딩 코사인(생성 vs x_ai). 의미보존. (단독으론 붕괴 과소평가 — rule과 함께 gemma 필요.)

---

## 8. 주의·한계 (애매함 제거)

1. **모든 Stage3 실험 = 1,561쌍(n_sample=1)·옛 D.** `dpo_pairs_eos_n4.jsonl`(4,605, n_sample=4)는 **미학습**. 채택안(DPOP λ5)을 n4로 본학습 필요.
2. **D2로 바꿔도 붕괴 못 막음.** 실측: D2도 붕괴 DPO 출력을 human(0.009)으로 봄. detector는 붕괴탐지기가 아니라 AI/사람 분류기. 게다가 D2 hard-neg율(3%)이 옛D(23%)보다 낮아 데이터도 덜 나옴.
3. **강회피(D<0.1)는 붕괴 필연** — DPOP λ1(57%)·RPO(52%) 동일. 깨끗한 강회피 원하면 **Stage4 refinement**(붕괴 문장 사후 교정) 필요(논문도 이 단계 있음).
4. rule/cos만 보면 성공으로 오판(RPO rule 8%·cos 0.968인데 gemma 52%). **판정은 반드시 plain beam4 + gemma.**

---

## 9. 다음 단계

- [ ] **DPOP λ5를 n4(4,605쌍)로 본학습** → held-out(plain beam4 + gemma) 재확인.
- [ ] (선택) λ 스윕(2~4)으로 "깨끗 유지 최대 회피" 지점 미세조정.
- [ ] (강회피 원할 시) DPOP λ1/RPO + **Stage4 refinement** 구현.
- [ ] 최종 보고는 **plain beam4 + gemma** 기준으로만.
