# pairs_v1 — 한국어 인간↔AI 논문초록 pair 데이터셋

**5,553쌍** · 2026-08-12 · MASH(ACL 2026 Findings) Stage 1 방식

인간이 쓴 KCI 논문 초록과, 같은 내용을 LLM이 다시 쓴 초록의 짝이다.
두 글 모두 카피킬러로 검사해 **인간 글은 인간으로, AI 글은 AI로 판정된 것만** 남겼다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `pairs_v1.jsonl` | **최종 데이터셋 5,553쌍** |
| `human_pool.jsonl` | 인간 초록 원본 풀 9,425편 (수집 단계 산출물) |
| `gen_full_p3b_clean.jsonl` | AI 재서술 9,115편 (깨짐 검사 통과분) |
| `copykiller_full.jsonl` | 카피킬러 점수 18,390건 (원시) |
| `kci_eval_bodies.jsonl` | 평가셋용 본문+초록 300편 (학습셋과 분리) |

## 레코드

```json
{
  "id": "abstract-ART000850832",
  "human_text": "…", "ai_text": "…",
  "generator": "Qwen3-8B-Q4_K_M", "prompt_version": "P3b",
  "research_field": "FIELD:B210000", "journal": "000485",
  "publication_date": "20060125", "original_title": "…",
  "kci_article_id": "ART000850832",
  "detector": "copykiller", "ck_human_pct": 0, "ck_ai_pct": 100,
  "split": "train"
}
```

`ck_human_pct` / `ck_ai_pct` 는 카피킬러 AI작성률(%)이다. 임계값 50%(논문 τ=0.5) 기준으로
인간은 미만, AI는 이상인 것만 들어 있다.

## 어떻게 만들었나

```
KCI 논문 초록 9,425편  (공공데이터포털 KCI 논문정보서비스 API, 2005~2020년)
   │  Qwen3-8B (Q4_K_M) · 프롬프트 P3b · temperature 0.8 · seed 42
   ▼
AI 초록 9,425편 → 깨짐 검사 → 9,115편
   │  카피킬러 전수 검사 (18,390건)
   ▼
5,553쌍
```

- 원문 1편당 AI 쌍 1개 (**1:1**) — 논문 Stage 1 과 동일
- 인간 원문은 **2021년 이전**만 사용 (ChatGPT 출시 이전 보장)
- 프롬프트 전문과 선정 근거: [`prompt_p3b.md`](prompt_p3b.md)

## 수율

| 단계 | 통과 | 비율 |
| --- | --- | --- |
| 생성 (깨짐 검사) | 9,115 / 9,425 | 96.7% |
| 인간 원문이 인간으로 판정 | 8,593 / 9,425 | **91.2%** |
| AI 초록이 AI로 판정 (flip) | 6,367 / 9,115 | **69.9%** |
| **최종 유효쌍** | **5,553 / 9,425** | **58.9%** |

**카피킬러의 한국어 학술 초록 오탐률 = 8.8%** (n=9,425, 95% CI 8.3~9.4%).
대상은 2005~2020년 인간 저작 초록이다. 무하유가 공개하지 않는 수치다.

## 분할

`split` 필드에 train 4,442 / dev 555 / test 556 (8:1:1).
**원문 단위로 나눴다** — 지금은 1:1이라 쌍 단위와 같지만, 나중에 한 원문에서 쌍을
여러 개 만들더라도 같은 원문이 서로 다른 split 에 흩어지지 않는다.

## 쓸 때 주의

- **도메인이 한국어 논문 초록에 한정**된다. 다른 장르로 전이될지는 미검증.
- **탐지기 하나(카피킬러) 기준**이다. 다른 탐지기에서도 같은 라벨이 나올지는 별도 검증이 필요하다.
- 인간 원문 중 약 1%에 서지 정보(이메일·목차·접수일·페이지번호)가 섞여 있다.
  카피킬러 판정에는 영향이 없었으나(오염분 오탐률 5% < 전체 8.8%), 학습 전 한 번 훑어볼 것.
- **저작권**: KCI 초록 원문이 그대로 들어 있다. 프로젝트 내부 공유용이며 재배포하지 말 것.

## 재현

코드: <https://github.com/sybyrr/AIKU-LLM-Text-Humanization>
`scripts/collect_kci.py` → `stage0_build_prompts.py` → `generate.py` →
`stage0_export_for_detector.py` → `stage0_analyze_copykiller.py` → `build_final_dataset.py`
