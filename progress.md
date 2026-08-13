# progress.md — 진행 현황과 다음 할 일

> 갱신: **2026-08-11** (폴더 인수 + 정리 + MASH v0 인수). 이 파일이 프로젝트 **상태의 단일 기준**이다.
> 연구 내용·의사결정 근거는 [notes/](notes/README.md), 폴더 구조·재현 절차는 [README.md](README.md) — 여기에 중복 기재하지 않는다.

**현재 1차 목표**: 두 트랙(Russell·MASH)의 인간↔AI pair를 탐지기(**카피킬러 + Pangram** 예정)에 넣어 AI→AI, 인간→인간으로 판별되는지 확인 — 데이터 완전성 검증. 이후 MASH 방식으로 LLM 튜닝.

## 한눈에: 파이프라인 현황 (Russell 트랙)

| # | 단계 | 상태 | 산출물 | 상세 |
| --- | --- | --- | --- | --- |
| 0 | 코퍼스 확보 | ✅ (주의 1건) | `corpus/raw/` 공식 zip 2개 | [notes/40](notes/40-코퍼스-구조-용어.md) · ⚠️ 아래 B-6 판본 불일치 |
| 1 | 인간 풀 100편 | ✅ | `dataset/human/full_base.jsonl` | [notes/41 §1](notes/41-데이터셋-구축-작업보고.md) |
| 2 | 프롬프트 A/C/E ×100 | ✅ | `dataset/prompts/prompts_full.jsonl` (300) | [notes/41 §3–4](notes/41-데이터셋-구축-작업보고.md), [notes/42 §3](notes/42-생성-명세와-결과.md) |
| 3 | 생성 9모델 ×300 | ✅ | `dataset/generations/` — 클리닝 후 **8모델 2,400건** | [notes/42 §6](notes/42-생성-명세와-결과.md) + 아래 "노트 미기록 작업" |
| 4a | 검증 — Fast-DetectGPT | ⚠️ **구세트 대상** | `dataset/scores/detect_scores.jsonl` (2,199) | 아래 A-1 |
| 4b | 검증 — LLM judge (Russell) | ❌ **중단, 출력물 없음** | `logs/judge_*` 만 존재 | 아래 A-2 |
| 4c | 검증 — 카피킬러 · Pangram · 사람 평가 | ❌ 미착수 | — | [notes/02 TODO③](notes/02-회의록-2차-0806.md) |

핵심 수치(2026-08-11 기준): Fast-DetectGPT TPR@FPR5% (임계 0.126) 전체 56.2%, 모델별 gemma-3-27b 87% ~ qwen3-8b 31% — 즉 **탐지기 하나 기준으로도 절반 가까이가 '사람'으로 통과**하는 상태로, 조건(A/C/E) 효과는 ≤5%p, 모델 패밀리 효과가 지배적. 상세 표는 [notes/42 §6-D](notes/42-생성-명세와-결과.md).

## MASH 트랙 현황 (`mash/` — 2026-08-11 김용재에게서 v0 인수)

| 단계 | 상태 | 비고 |
| --- | --- | --- |
| 인간 코퍼스 (KCI 논문 초록) | ✅ 1,497편 — **6,000쌍에는 부족** ([확장 경로](mash/corpus_expansion.md)) | 5,000 수집 → 정제 2,194 → journal당 100건 cap 샘플링(seed=42). 한국어 초록 검증(한글비 ≥0.30), 2021년 이전, 분야 67·저널 167, 중앙값 607자. 상세: `mash/kci_humanization_dataset_v0/README.md` |
| pair 생성 파이프라인 | ✅ 구현됨 (코드는 이 사본에 없음) | 프롬프트 v3 = 문장 단위 충실 paraphrase (`mash/…/prompt_v3.txt`) |
| AI pair 생성 | ✅ **본 구축 완료** — 9,425건 생성(오류 0) | 결과·해석: **[notes/43](notes/43-MASH-파일럿-Stage0.md)** |
| **최종 데이터셋** | ✅ **5,553쌍** — `mash/pairs_v1.jsonl` | 명세: [mash/DATASET.md](mash/DATASET.md) |
| 프롬프트 최적화 | ✅ **P3b 확정** (원문당 0.64쌍, 복사율 16.4%) | 5개 변형 비교 완료. 근거: [notes/43 결과 ⑧·⑨](notes/43-MASH-파일럿-Stage0.md) |
| 모델 선정 | ✅ **qwen3-8b** | EXAONE-3.5-7.8B 탈락 — 초록을 번호 목록으로 재구조화(마크다운 31/50). qwen은 0/150 |
| 탐지기 검증 | ✅ **카피킬러 18,390건 전수 완료** | Fast-DetectGPT는 이 도메인에서 사용 불가(판정이 정반대) |

**Stage 0 결론: 🟢 GO — 파일럿 종료** (카피킬러 500건, [notes/43](notes/43-MASH-파일럿-Stage0.md))
① **복사율이 flip을 결정한다** — 같은 모델·같은 원문에서 12자 겹침 16→49%로 훑으니 AI판정률 80→46% 단조 감소, **r = −0.945**. 프롬프트는 복사율을 통해서만 작용한다(EXAONE 대조군에서 확인: 복사율이 같으면 P1·P2 차이가 사라진다)
② **인간 오탐률 13.3%** (n=150, 95% CI 8.8~19.7%) — 무하유 미공개 수치, 독립 측정으로 보고 가치 있음. 동시에 수율의 상한
③ **채택: qwen3-8b × P3b** — 유효쌍 38%, 복사율 16.4%(P1의 절반). P1과 수율 차는 유의하지 않으나 복사율이 낮아 학습 신호가 건강하다
④ ⚠️ **탐지기가 결론을 뒤집는다**: 같은 텍스트에 Fast-DetectGPT는 flip 6~14%, 카피킬러는 46~96%. 필터 탐지기 = 공격 대상 탐지기 원칙을 지켜야 한다
⑤ **1:1 구성 · P3b 단독**으로 원문당 0.64쌍 → 현행 1,497편이면 **약 960쌍**(archive/handoff_20260811.md 1차 목표 500~1,000 달성). **6,000쌍에는 9,375편 필요** — 코퍼스 확장이 유일한 병목

**도메인 타당성 (2026-08-12 재검토 — 종전 서술 정정).** "논문에 학술 도메인이 없다"는 종전 메모는 틀렸다.
MGT-Academic의 **STEM = ArXiv 과학·기술 논문**이고, 논문 Table 1에서 **STEM이 ASR 1.00 으로 6개 도메인 중 최고**다
(Social 0.98 · Essay 0.95 · WP 0.90 · Humanity 0.87 · **Reuters(뉴스) 0.73 최저**). 학술 산문은 MASH가 가장 잘 통한 영역이고,
우리가 러셀 트랙에서 겪은 뉴스 도메인의 어려움도 논문과 방향이 같다. → **초록 도메인 유지.**
KatFish 논증문(~470편)은 1:1로 6,000쌍에 필요한 9,375편의 5%라 대체재가 못 된다.

⚠️ 다만 논문이 명시한 전제 하나가 우리에게 걸린다 (Limitations, `MASH.md:367`):
"evasion is only mathematically meaningful against detectors that maintain a **reasonable FPR**."
카피킬러의 한국어 초록 오탐률 13.3%는 이 전제의 경계선이다 — 통과(86.7%)는 하지만 여유가 크지 않다.
Figure 5 ablation에서도 "대부분 machine으로 분류되는" 시드 코퍼스는 초기화에 부적합했다.

## 공개 범위 — GitHub(공개) vs Google Drive(팀 내부)

2026-08-12 결정: 저장소는 **공개**, 데이터는 **Drive 로 소수 참여자에게만**. `.gitignore` 가 이 경계를 강제한다.

| | 올라가는 것 | 이유 |
| --- | --- | --- |
| **GitHub (공개)** | `scripts/` · `notes/` · `logs/` · `mash/*.md` · `archive/`(문서만) · 루트 문서 — **106개 파일, 6.1MB** | 코드와 연구 기록. 코퍼스 텍스트 0건 |
| **Google Drive (비공개)** | 인간 풀·생성 pair·프롬프트·docx 배치·카피킬러 PDF·검수 HTML | 저작권·약관상 재배포 불가 |

**커밋에서 반드시 빠져야 하는 것** (검사로 확인 완료):
- `.API_KEY` · `.kli_api_key` — 키 문자열이 트리 어디에도 새지 않았음을 전수 확인
- `archive/claude-home-backup/` — **사용자 이메일 + 세션 대화 전문**이 들어 있다 (494파일)
- `archive/aihub_essay_2024_raw/` — AI Hub 약관: 승인받지 않은 제3자 열람 금지
- 모든 `*.jsonl` — **프롬프트 파일도 포함**. 조건 B 는 기사 도입부 2문장, A 는 제목, C 는 요약문을 원문 그대로 담는다
- `archive/pilot/*.json` · `review/*.html` — 인간 원문·AI 재서술 전문이 박혀 있다

⚠️ **gitignore 는 줄 끝 주석을 지원하지 않는다.** `models/  # 9GB` 로 쓰면 전체가 패턴이 되어 아무것도 안 걸린다.
⚠️ 규칙을 고친 뒤에는 `rm -f .git/index && git add -A` 로 인덱스를 다시 만들어야 한다 — `git rm --cached` 는 조용히 실패한다.

**미결**: `archive/notion-export/` 와 `notes/01·02`(회의록)에 팀원 실명이 들어 있다(진민재 9 · 송성준 12 · 김용재 10개 파일).
공개 저장소에 올리려면 **본인 동의가 필요**하다. 동의 전에는 커밋하지 않는다.

## 작업 타임라인 (요약 — 수치·근거는 notes/41·42)

| 날짜 (2026) | 작업 | 기록 |
| --- | --- | --- |
| 07-30 | 1차 회의: 두 방법론 갈래, 방향별 RQ | [notes/01](notes/01-회의록-1차-0730.md) |
| 08-01 | 연구 프레이밍 5편 (지형·RQ·특징·개입지점·평가) — 송성준 | [notes/10–21](notes/README.md) |
| 08-03 | MASH 병렬 트랙 제안 — 진민재 | [notes/30](notes/30-MASH-기반-제안.md) |
| 08-06 | 2차 회의: 역할 분담, TODO(~8/12) | [notes/02](notes/02-회의록-2차-0806.md) |
| ~08-07 | 파일럿 2회 (10편×3조건×3모델): 조건 B 폐기→E 대체, 페르소나 프롬프트·temp 0.0 채택, 버그 3건 수정 | [notes/42 §4–6](notes/42-생성-명세와-결과.md) · `archive/pilot/` |
| 08-08 | 본 실행 100편×3조건×5모델 (17:50–20:57, 노트의 "2022"는 오타) + 야간 gemma·qwen3-30b-a3b·양자화 비교 | [notes/42 §6-B/C](notes/42-생성-명세와-결과.md) |
| 08-08~11 (로그 무일자) | 국산 모델 2종 추가 → 클리닝 → Fast-DetectGPT 채점(버그 수정 재실행) → judge 시도(중단) | 아래 "노트 미기록 작업" |
| 08-11 | 스냅샷 아카이브(`archive/*_20260811.jsonl`) 후 폴더 이관 (git 이력 없음, mtime 전부 리셋) | — |
| 08-11 | MASH 트랙 v0 인수(`mash/`), 탐지기 방향 확정(카피킬러+Pangram), MASH 생성 모델 상향 결정 | 위 "MASH 트랙 현황" |
| 08-11 야간 | Stage 0 파일럿 완료 → 카피킬러 검증 → **GO**. EXAONE 교차 확인, P3 프롬프트 3종 설계, AI Hub 신규 데이터 판정, 코퍼스 확장 경로 조사 | [notes/43](notes/43-MASH-파일럿-Stage0.md) · [mash/aihub_verdict.md](mash/aihub_verdict.md) · [mash/corpus_expansion.md](mash/corpus_expansion.md) |

## 노트에 기록되지 않은 작업 (파일시스템에서 복원 — 현재 이 절이 유일한 기록)

notes/41·42는 7모델 시점까지만 기록하고 있다. 그 이후:

1. **국산 모델 2종 추가** — kanana-8b(Kakao), midm-11b(KT)를 HF→F16 GGUF 변환·Q4_K_M 양자화(각 ~2.6분, 성공) 후 300건씩 생성 (18m53s / 22m00s, 오류 0). 목적 추정: EXAONE 특이 현상 vs 한국어 모델 공통 현상 분리. 로그: `logs/convert_*`, `logs/quant_*`, `logs/server_808{0,1}.log`.
2. **클리닝 2단계** (스냅샷: `archive/`) — ① exaone-7.8b **제외**: 300/300 마크다운 유출(`archive/exaone-7.8b_dropped.jsonl`), ② kanana-8b 전 300건 말미의 `<|eot_id|>` 문자열 제거(원본: `archive/kanana-8b_eot_raw.jsonl`). 9모델 2,700건(`full_generations_9models_20260811.jsonl`) → 현행 8모델 2,400건. 스냅샷 간 관계는 레코드 단위로 검증됨.
3. **LLM judge 시도 (실패로 종료)** — qwen3-32b Q4_K_M, 포트 8090(slot당 ctx 8192), variant `zero_shot_cot`, 사고모드 on, 대상 인간 100+AI 200(조건 A). **216/300 처리 시점에 중단**(로그상 오류 0 = 수동 중단 추정, ~103분, 7–8 tok/s). `detect_llm_judge.py`는 완주해야만 --out을 쓰므로 **결과 파일이 없다**. 래퍼 스크립트도 없음(run 기록은 `logs/judge_run_cot_A.log` 74바이트 라벨뿐).
4. **검수 페이지** — `review/full_review.html`+`full_cards.json`은 7모델×3문서 시점의 구버전이며, full_cards.json을 만드는 duckdb 단계는 저장소에 없다.

## 다음 할 일

### M. MASH 트랙 — 본 구축 진입 (2026-08-12)

**Stage 0 파일럿 종료.** 확정 사항 (2026-08-12 사용자 결정):
- **구성: 1:1** — 원문 1편당 AI pair 1개. 논문 Stage 1과 동일(`MASH.md:102`, "for each human sample … **a** paraphrase"). 여러 arm으로 원문당 여러 쌍을 만들면 SFT의 목표 문장·DPO의 chosen response가 중복돼 실효 다양성이 원문 수로 수렴한다.
- **arm: qwen3-8b × P3b 단독.** P1을 붙이면 커버리지 32→36/43(+12.5%)이나 생성·검사량이 2배가 된다. P3a·P3c·P2는 커버리지를 1편도 못 늘린다(P3b가 지배).
- **게이트: 보존 게이트 폐기 → "깨짐" 검사만.** 논문은 $D(x_{human})<\tau$, $D(x_{ai})>\tau$ 두 조건뿐이고 내용 보존 필터가 없다. BERTScore·GRUEN은 최종 공격 산출물 평가 지표이지 데이터 필터가 아니다. 실측상 qwen 생성물의 깨짐률은 **250건 중 2건(0.8%)**.
- **Pangram 보류.** 카피킬러 전량 검사로 간다(사용자가 docx 배치 업로드 감당 가능).

**규모 산술 (1:1 · P3b 단독)**: 원문 1편당 유효쌍 **0.64** (인간측 통과 86% × flip 74% × 깨짐 통과 99%)
- 6,000쌍 → **인간 초록 9,375편 필요** (현재 1,497편, **약 7,900편 부족**)
- 검사 대상 18,750건 = 카피킬러 **54회**
- 현행 1,497편만으로 **약 960쌍** — Stage 2 SFT 착수는 지금도 가능

**판단이 필요한 것 (사용자)**
1. **KCI 인증키 발급 + IP 차단 해제** — 아래 § 코퍼스 확장 참조. 이게 6,000쌍의 유일한 병목이다.
2. **960쌍으로 Stage 2 SFT를 먼저 돌려볼지, 코퍼스를 기다릴지**
3. **받은 AI Hub 데이터는 사용 불가 판정** — [mash/aihub_verdict.md](mash/aihub_verdict.md). 2024~2025년 수집분이고 18%에 LLM 문단이 삽입돼 있다.

### 코퍼스 확장 — KCI Open API (2026-08-12 조사)

**서버에서 실측한 접근성** (`curl`):
- `www.kci.go.kr` → **HTTP 400, 우리 IP 163.152.26.133 방화벽 차단** (차단 안내가 `TEL 042-869-6738` 명시)
- `apis.data.go.kr` → **정상 응답** (게이트웨이가 `NO_OPENAPI_SERVICE_ERROR` 반환 = 도달 가능)
- `www.data.go.kr` → 200

**채택 경로: [data.go.kr 15085348](https://www.data.go.kr/data/15085348/openapi.do) (KCI 논문정보서비스).** 사용자 결정 — IP 차단 해제는 요청하지 않는다(학부 프로젝트라 KCI와 협의할 유인이 없음).

| 경로 | 판정 |
| --- | --- |
| **data.go.kr 15085348** | ✅ **채택.** `apis.data.go.kr/B552540/KCIOpenApi/artiInfo` 에 실제 호스팅 — 서버에서 도달 가능. 상세기능 20종, 개발계정 **자동승인·5,000건/일**(요청당 100건 → 9,375편에 94회 호출이면 충분, 운영계정 심의 불필요). ❓초록 제공 여부는 공개 문서에 미명시 — **키 수령 후 오퍼레이션별로 실측 확정 필요** |
| data.go.kr 3049042 | ❌ "링크형" — kci.go.kr 로 넘기는 껍데기. 그 호스트가 차단돼 있어 무용 |
| KCI 포털 Open API | ❌ 초록은 확실하나(`abstract(articleInfo)`) **IP 차단**. 해제 요청 안 하기로 결정 |

같이 신청: [15084891 KCI_공용코드](https://www.data.go.kr/data/15084891/openapi.do) (연구분야 코드표 — 분야 필터에 필요)

인증키는 `/workspace/.data_go_kr_key` 에 보관(gitignore 대상, `.kli_api_key` 와 동일 취급 — 열람·출력·커밋 금지).

**보류 중인 아이디어 — RoBERTa 두 번째 탐지기** (2026-08-12 제안, **실행 안 함**)
논문은 벤치마크를 3:1:1:1 로 나눠 RoBERTa 를 도메인별 이진 분류기로 파인튜닝했다(`MASH.md:522·526`).
우리는 카피킬러를 그대로 쓰므로 필수가 아니지만, 붙이면 ① 논문과 직접 비교 ② "탐지기 하나에만 맞췄다"는 한계 해소
③ 재현 가능한 공개 탐지기 확보가 된다. 착수 조건과 비용:
- **기존 데이터 재사용 불가.** ⓐ $\mathcal{D}_{pair}$ 로 학습하면 탐지기가 $y_w$ 를 외워 순환이 된다 —
  반드시 분리 수집. ⓑ 우리 $x_{ai}$ 는 전부 **재서술**이라, 그걸로 학습하면 "처음부터 쓴 AI 초록"을 못 잡는다.
- 필요량은 작다(논문 추정 도메인당 기계 300 + 인간 300). 인간 1,000~2,000편 = API 15분,
  AI 1,000~2,000편 = **본문/제목 → 초록 생성** 2~4시간. 평가용 본문 300편이 이미 그 성격이라 그대로 쓰인다.
- **카피킬러 검증이 끝난 뒤 판단한다.** 지금 착수하면 본 파이프라인과 GPU·시간이 겹친다.

**그다음 (내가 진행 가능 — 승인 후 착수)**
5. **본 구축**: KCI 1,497편 × arm 3개 = 4,491건 생성 (qwen3-8b, ~7시간) → 보존 QA → 카피킬러 배치 13회분 내보내기
   ⚠️ 카피킬러는 배치당 350개 수동 업로드다. 4,491건이면 **13회 업로드**가 필요하다 — 이 병목을 사용자와 먼저 상의할 것(전량 검사 대신 표본 검사로 갈지)
6. 스키마는 `mash/…/README.md` § Pair 주요 field 준수. **원문 단위로 train/dev/test를 나눌 것** — 같은 원문에서 나온 여러 arm의 쌍은 독립이 아니다
7. Pangram 도입 시 같은 500건을 재채점하면 탐지기 간 일치도를 얻을 수 있다(추가 생성 불필요)

### V. 탐지기 검증 (두 트랙 공통 1차 목표)
6. **⚠️ 코퍼스 업로드 약관 판단** — 모두의 말뭉치 "사전 승낙 없는 전송" 금지(제12조), AI Hub "승인받지 않은 제3자 열람 금지 + 국외 반출 별도 합의". 반대로 카피킬러 약관 제14조는 업로더에게 처분권한 보증을 요구하고 제7조는 원문을 일정 기간 서버 저장한다. → **우리가 저작권을 갖는 AI 생성문 먼저, 인간 원문은 승인 후**의 단계적 진행 권장. (KCI 초록은 공개 학술 초록이라 상대적으로 위험 낮음 — 판단 필요)
7. **카피킬러 실무 사양** — 다중 업로드 **350개·총 200MB** 가능(폴더/ZIP 미지원 추정), **공개 API 없음**, 출력은 PDF 확인서 4종뿐이고 원시 확률 미노출. "AI작성률"=약 400자 검사 단위의 **어절 기반 비율**. → 정량 지표로는 부적합, **국내 실무 표준과의 대조** 용도.
8. **Pangram 도입 검토** — 공식 API·파이썬 SDK(`pangram-sdk`)·대량 처리 지원. 한국어 수치 공개(FPR 0.0000%/FNR 0.58%, 기술보고서 Table 8)이나 **웹 크롤 텍스트 기준이라 한국어 신문·초록 FPR은 미보고** → 우리 측정이 신규 기여. 결과 보존 48시간.
9. **Russell 생성물 중 최선 모델 선별 → 카피킬러·Pangram 검사** (기존 8모델 재채점 A-1 선행).

### A. Russell 트랙 — 기존 검증 마무리 (중단 지점에서 재개)
1. **Fast-DetectGPT 재실행** — 현행 8모델 2,400건 대상. 기존 `detect_scores.jsonl`은 구 7모델 세트 기준: 드롭된 exaone-7.8b 점수 300건 포함, kanana-8b·midm-11b **600건 미채점**. (기존 점수의 모집단은 `archive/full_generations_9models_20260811.jsonl` — 그래서 스냅샷 삭제 금지)
2. **LLM judge 재실행** — `run_judge.sh` 작성 권장(포트·ctx 설정 포함). 처리량이 병목(사고모드 32B ~7-8 tok/s → 300편 ≈ 2.4h): `--think` off / max_tokens 축소 / 경량 judge 검토. 4개 variant 중 zero_shot·zero_shot_cot가 언어독립 기준선.
3. **미결 생성 1건** — qwen3-30b-a3b / 조건 C / `NPRW2300000008.21937`: 결정적 HTTP 500 (4시드 실패). `--reasoning-format none` 등 서버 옵션 시도, 또는 영구 결번으로 문서화 (오류 레코드는 스키마가 다름: `{cond,doc_id,error,model}` — 다운스트림 코드가 견뎌야 함).
4. quant arm(qwen3-8b q4/q8, 조건 A만) — 산출물에 포함할지, 채점할지 결정.
5. 요약 1건 copy 초과(`NIRW2300000002.22283`, max_copy 26>25) — 재생성 또는 예외 문서화.
6. 검수 페이지 재생성 — full_cards.json 생성 duckdb 쿼리를 스크립트로 복원 후 8모델×100문서 기준으로.

### B. 데이터셋 신뢰성 (notes/41 §8 "남은 일" 승계)
7. **코퍼스 판본 불일치 해소** ⚠️ — `corpus/raw/` zip은 22-시리즈(신문 말뭉치 2022)인데 데이터셋 doc_id는 23-시리즈(2023판, HF 미러 유래). **2023년판 공식본을 받아야** 풀 재현·공식본 검증([notes/40 §6](notes/40-코퍼스-구조-용어.md))이 가능. 미러 parquet도 미복사(`corpus/mirror/README.md`).
8. 의미 정합 측정 — 엔티티/수치 일치 1차 + BGE-m3 하한 스크리닝 ([notes/41 §8](notes/41-데이터셋-구축-작업보고.md)).
9. 모델의 2022 기사 암기 측정 — ≥5-gram 일치를 인간-인간 중복 분포와 비교.
10. HF 미러 재배포 조건 확인 (extract_russell_pool.sql·make_review_html.sh에 미확인 플래그).

### C. 사람 평가 준비 ([notes/02 TODO③](notes/02-회의록-2차-0806.md), [notes/21 §4](notes/21-러셀-ACL2025-평가방식.md))
11. 팀 외부 평가자 모집 설계 (2단계 스크리닝, 라벨+확신도+하이라이트+자유 설명 4종 수집), 카피킬러·Pangram 실행 방안 (카피킬러는 API 없음 — [notes/README 쟁점 3](notes/README.md)).

### D. 전략 결정 필요 (회의 안건 후보 — [notes/README "열려 있는 쟁점"](notes/README.md))
- 신호원: 사람 판정 vs 탐지기 점수 (13 vs 30 문서의 충돌) · 도메인 확정 · 데이터 스케일 · Qwen/Gemma 길이 미달 모델의 구제 여부([notes/42 §6-B](notes/42-생성-명세와-결과.md) "미결") · exaone-7.8b 재등판 여부(가중치는 `models/`에 있음).

## 알려진 문제 (스크립트/환경) — 재실행 전 필독

- ~~이 사본은 그대로는 실행 불가~~ → **2026-08-11 해소.** 현재 실행 환경:
  - **GPU**: 호스트에 TITAN Xp 7장. `cont1`(Claude 세션, GPU 없음) + `gpu1`(GPU 4·5 할당, `--network container:cont1`로 네트워크 공유) 2컨테이너 구성. GPU 0~3은 **타인 사용 중** — 건드리지 말 것. 서버 기동: `docker exec -d gpu1 bash /workspace/scripts/gpu_serve.sh <gguf> 8080` (호스트에서 실행, 재부팅 후 수동 재기동 필요)
  - **llama.cpp**: CUDA 빌드는 `gpu1:/home/dev/llama.cpp/build/bin/` (볼륨 `ssj` 마운트, 이미지 `ssj-built`). CPU 빌드는 `/workspace/tools/llama-b10360/` (cont1 전용 — 다른 컨테이너에선 libgomp 부재로 실행 불가)
  - **파이썬**: 반드시 `/workspace/.venv/bin/python3` 를 쓴다. 맨 `python3` 는 `/opt/venv` 가 잡혀 **kiwipiepy·torch·docx가 없다** — `stage0_check_pairs.py` 는 이때 경고만 내고 고유명사 검사를 건너뛰어 보존율이 부풀려진다(실측 66%→74%). torch 2.13/cu126(CPU 폴백), transformers 5.15
  - **모델**: `models/`에 EXAONE-3.5-7.8B + Qwen3-8B(신규 다운로드). 나머지 7종은 여전히 부재 → HF에서 재확보 필요
  - 여전히 부재: duckdb (프롬프트 조립 SQL 실행 불가 — 파이썬 대체 필요)
- `generate.py --url` 은 **전체 엔드포인트**를 요구한다(`http://127.0.0.1:8081/v1/chat/completions`). 호스트만 주면 전건 HTTP 404로 조용히 실패한다.
- `gpu_serve.sh` 의 중복 기동 방지는 **포트만 보고 모델은 보지 않는다.** 다른 모델을 같은 포트로 올리려 하면 "이미 서버가 떠 있다"며 무시한다 — 두 모델 동시 운용 시 포트를 반드시 나눌 것(8080/8081).
- **컨테이너 로케일이 POSIX** (`LANG` 비어 있음). tmux 는 이걸 보고 클라이언트를 비-UTF-8 로 잡아 한글을 글자당 `__` 로 깨뜨린다. `scripts/claude_session.sh` 가 `LANG=C.UTF-8` export + `tmux -u` 로 해결(2026-08-12). 수동 접속 시엔 `docker exec -it cont1 tmux -u attach -d -t claude`.
- 러너 3종(run_full/night/domestic.sh)은 `cd /workspace` 후 bare 파일명으로 .py 호출 → 이전 정리에서 스크립트가 `scripts/`로 이동해 **경로 수정 필요**. run_pilot.sh·make_pilot_html.sh는 상대경로 입력이 `archive/pilot/`으로 이동해 깨짐.
- `build_prompts_AEC.sql`은 파일럿 입력을 읽어 **운영 출력(`prompts_full.jsonl`)을 덮어씀** — 실행 금지, 이름에 pilot 명시 권장.
- 합본 cat이 자기 출력 포함(`generations/*.jsonl` glob에 full_generations.jsonl 매치) → 통계 이중 집계. 합본 위치 분리 또는 glob 수정 필요.
- run_night.sh는 호출자가 run_full.sh 출력을 `/tmp/full.log`로 리다이렉트해야 완료 감지 가능.
- Dockerfile은 `/opt/venv`(torch 2.5.1/cu121)를 만들지만 실제 실행은 `/workspace/.venv`(torch 2.13/cu126)였음 — 컨테이너 레시피와 실환경 불일치.
- GPU 제약(원 머신): TITAN Xp sm_61 ×4 → CUDA 13·vLLM·FlashAttention 불가, llama.cpp 소스 빌드 필수 ([notes/41 §2](notes/41-데이터셋-구축-작업보고.md)).

## 정리 이력

### 2026-08-11 — 폴더 인수 정리 (이 세션)
- 삭제: 루트 `EXAONE-3.5-7.8B-Q4_K_M.gguf` 4.7GB (`models/` 사본과 md5 동일 `9a854faa…`, 참조 0건 확인)
- 개명: `.kli_api_key]` → `.kli_api_key` (download_corpus.py 기대 경로·.gitignore 규칙과 일치)
- 이동: `개인 페이지 & 공유된 페이지/` → `archive/notion-export/` — notes/와 내용 동일 검증(md 9건 정규화 diff 손실 0, png 5건 md5 동일). 유일한 고유 정보였던 팀원 Notion 프로필 URL 6건은 [notes/00](notes/00-프로젝트-개요.md)에 이관.
- 이동: `detect_scores_buggy.jsonl`(구버그 점수), `reorganize.sh`(완료된 1회성 정리 스크립트) → `archive/`
- 삭제: 빈 디렉토리 `archive/pilot/gen_full/`, `gen_quant/` · 개명: `dockerfile` → `Dockerfile`
- 신설: `CLAUDE.md`, `progress.md`, `corpus/mirror/README.md` · 보수: `notes/README.md` (색인에 40–42 추가, 원본 경로 갱신)
- 신설: `mash/` — 루트에 도착한 `kci_humanization_dataset_v0.zip`을 풀어 배치, 원본 zip도 같은 폴더에 보존

### 2026-08-11 이전 — 이관 전 정리 (원 작업자, `archive/reorganize.sh`)
- 플랫 `/workspace/data/`(53파일)를 현행 `scripts/ corpus/ dataset/ review/ archive/ logs/` 구조로 분리. 스크립트 내 상대경로 정리는 미완(위 "알려진 문제").
