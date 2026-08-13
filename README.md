# 한국어 LLM 생성문의 "AI 티" 억제 연구

한국어로 쓴 글이 AI 탐지기에 어떻게 걸리는지 재고, **인간 글 ↔ AI 글 쌍 데이터셋**을 만들어
탐지기의 신뢰성을 검증한다. 이후 이 데이터로 문체 변환 모델을 학습한다.

고려대 학부 프로젝트(AIKU)이며, 방법론은
[MASH (ACL 2026 Findings)](https://arxiv.org/abs/2601.08564)와 Russell et al. (ACL 2025)를 따른다.

## 두 트랙

| | Russell 트랙 | MASH 트랙 |
| --- | --- | --- |
| 방향 | 제목·요약을 주고 **AI가 새로 씀** | 인간 글을 **AI가 다시 씀** |
| 인간 원문 | 신문기사 100편 (국립국어원 모두의 말뭉치) | KCI 논문 초록 9,425편 |
| 현재 | 8모델 × 300건 생성 완료, 탐지기 재검증 대기 | ✅ **5,553쌍 완성** |

## 주요 발견

**① 탐지기를 바꾸면 결론이 뒤집힌다.** 같은 텍스트 500건에 대해
Fast-DetectGPT는 label flip 6~14%, 카피킬러는 46~96%로 판정했다. 두 결과는 사실상 무관하다.
→ MASH가 요구한 "필터로 쓰는 탐지기 = 공격 대상 탐지기" 원칙이 실측으로 확인됐다.

**② 원문을 많이 베낄수록 AI로 안 잡힌다.** 12자 n그램 겹침이 0~10%인 구간의 AI 판정률은 80.4%,
50% 이상인 구간은 23.8%로 **단조 감소**한다 (**r = −0.990**, n=9,115).
"충실하게 재서술하라"는 지시가 오히려 복사를 유도해 파이프라인을 망가뜨리고 있었다.

**③ 카피킬러의 한국어 학술 초록 오탐률 8.8%** (n=9,425, 95% CI 8.3~9.4%).
대상은 2005~2020년, 즉 ChatGPT 이전이 확정된 인간 글이다. 무하유는 이 수치를 공개하지 않는다.

| 탐지기 지표 (카피킬러, 임계 50%) | |
| --- | --- |
| 위양성률 FPR / 위음성률 FNR | 8.8% / 30.1% |
| 정밀도 / 재현율 / F1 | 88.4% / 69.9% / 78.1% |

⚠️ 이 재현율을 탐지기 성능으로 읽으면 안 된다 — 우리 AI 글은 인간 글의 *재서술*이라
원문의 인간다움을 물려받는다. 상세: [`mash/PIPELINE.md`](mash/PIPELINE.md)

## 저장소 구성

```
scripts/   수집 → 프롬프트 조립 → 생성 → 깨짐 검사 → 탐지기 내보내기 → 점수 파싱 → 최종 조립
notes/     연구 노트 16편 (진입점: notes/README.md)
mash/      MASH 트랙 — 데이터셋 명세 · 파이프라인 규격 · 확정 프롬프트
archive/   구버전 스냅샷 · 회의록 원본 (읽기 전용)
```

- 상태와 다음 할 일: [progress.md](progress.md) — **프로젝트 상태의 단일 기준**
- 데이터셋 명세: [mash/DATASET.md](mash/DATASET.md)
- 파이프라인 규격(타 도메인·탐지기 팀용): [mash/PIPELINE.md](mash/PIPELINE.md)
- 프롬프트 전문: [mash/prompt_p3b.md](mash/prompt_p3b.md)

---

# 재현 가이드

## 0. 준비

```bash
git clone https://github.com/sybyrr/AIKU-LLM-Text-Humanization.git
cd AIKU-LLM-Text-Humanization
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo apt install poppler-utils          # pdftotext — 카피킬러 PDF 파싱용
```

**데이터는 이 저장소에 없다.** 인간 원문·생성 pair·프롬프트·점수는 저작권과 약관 때문에
제외했다(국립국어원 모두의 말뭉치 제12조, KCI 초록 저작권, AI Hub 제3자 열람 금지).
프롬프트 파일도 마찬가지다 — 기사 제목·도입부·요약문을 원문 그대로 담기 때문이다.
데이터는 **프로젝트 참여자에게 Google Drive 로만 공유**한다. 받은 묶음을 이렇게 배치한다:

```
mash/pairs_v1.jsonl            mash/human_pool.jsonl        mash/seed_full.jsonl
mash/gen_full_p3b_clean.jsonl  mash/prompts_full_p3b.jsonl  mash/kci_eval_bodies.jsonl
pilot/scores/copykiller_scores.jsonl
```

## 1. 최종 데이터셋 재조립 — 그대로 됨

```bash
.venv/bin/python3 scripts/build_final_dataset.py --out mash/rebuilt.jsonl
```

**검증 완료**: Drive 묶음 + 저장소 코드만으로 5,553쌍이 바이트 단위로 동일하게 재생성된다.
게이트 임계값(`--tau 70`)이나 분할 비율(`--split 7:1:2`)을 바꿔 실험할 수 있다.

## 2. 처음부터 다시 만들기 — 본인 자원 필요

| 필요한 것 | 비고 |
| --- | --- |
| 공공데이터포털 인증키 | [KCI 논문정보서비스](https://www.data.go.kr/data/15085348/openapi.do) 활용신청(자동승인) → `.API_KEY` 에 저장 |
| GPU + `llama.cpp` + Qwen3-8B GGUF | 9,425건 생성에 약 9시간 (TITAN Xp 2장 기준) |
| 카피킬러 계정 | 18,390건 검사 · 3,500개씩 6회 업로드 |

```bash
.venv/bin/python3 scripts/collect_kci.py --out mash/human_ko.jsonl --target 8000
.venv/bin/python3 scripts/build_human_pool.py --in … --out mash/human_pool.jsonl
.venv/bin/python3 scripts/stage0_sample_pilot.py --in mash/human_pool.jsonl \
    --out mash/seed_full.jsonl --domain abstract --n 9425 \
    --text-field original_abstract --id-field kci_article_id
.venv/bin/python3 scripts/stage0_build_prompts.py --in mash/seed_full.jsonl \
    --out mash/prompts_full_p3b.jsonl --variants P3b
bash scripts/gpu_serve.sh <gguf> 8081 16384          # llama-server 기동
bash scripts/run_mash_full.sh                        # 생성 (--resume 내장)
.venv/bin/python3 scripts/stage0_export_for_detector.py --out batches \
    --human mash/seed_full.jsonl --gen mash/gen_full_p3b_clean.jsonl --batch-size 3500
#   → 폴더째 카피킬러 업로드 → 결과 PDF 저장
.venv/bin/python3 scripts/stage0_analyze_copykiller.py --pdf … --manifest … --out scores.jsonl
.venv/bin/python3 scripts/build_final_dataset.py --out mash/pairs_v2.jsonl
```

⚠️ `temperature 0.8` 이라 **seed 를 같게 줘도 서버 빌드·슬롯 수에 따라 생성물이 미세하게 달라진다.**
바이트 단위 재현이 필요하면 Drive 의 `gen_full_p3b_clean.jsonl` 을 쓸 것.

## 3. 다른 도메인 / 다른 탐지기 — 두 곳만 새로 쓰면 됨

```
                                      도메인 바꿀 때   탐지기 바꿀 때
 수집        collect_kci.py               ✏️ 새로 작성        —
 프롬프트     stage0_build_prompts.py      ✏️ 변형 추가         —
 생성        generate.py                     ✅ 그대로        ✅ 그대로
 깨짐 검사    stage0_check_pairs.py           ✅ 그대로        ✅ 그대로
 내보내기     stage0_export_for_detector.py   ✅ 그대로        △ 형식 확인
 점수 파싱    stage0_analyze_copykiller.py     —            ✏️ 새로 작성
 최종 조립    build_final_dataset.py          ✅ 그대로        ✅ 그대로
```

**도메인 교체** — 수집기 출력을 `human_pool.jsonl` 스키마에 맞추면 나머지는 그대로 돈다.

```json
{"kci_article_id": "<원문ID>", "original_abstract": "<본문>", "original_title": "…",
 "research_field": "…", "journal": "…", "publication_date": "…"}
```

**탐지기 교체** — 점수 파서 출력을 이 스키마로 맞춘다.

```json
{"file": "D00001.docx", "pair_id": "…", "label": "human|ai", "variant": "P3b",
 "model": "…", "ck_ai_pct": 0}
```

API 있는 탐지기(Pangram 등)라면 docx 내보내기를 건너뛰고 점수 jsonl 만 만들면 된다.
**게이트 3개는 도메인·탐지기가 달라져도 바꾸지 말 것** — 근거는 [mash/PIPELINE.md](mash/PIPELINE.md).

## 4. 환경

llama.cpp + GGUF(Q4_K_M) 로컬 서빙, TITAN Xp. 생성 모델은 Qwen3-8B.
학습 예정 모델은 KoBART(`gogamza/kobart-base-v2`, 124M) — 데이터셋 최대 660토큰으로
위치 임베딩 한계(1,026)의 64%만 쓴다.
