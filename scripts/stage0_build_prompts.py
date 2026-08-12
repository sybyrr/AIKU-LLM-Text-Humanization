#!/usr/bin/env python3
"""Stage 0 파일럿 — 인간 원문 → 재서술 프롬프트 조립.

generate.py 가 그대로 읽을 수 있는 스키마로 낸다: {doc_id, cond, target_char, prompt, system}
(generate.py 는 프롬프트를 입력 파일에서만 읽으므로 생성 코드 수정이 필요 없다.)

프롬프트 두 계열을 비교한다 — 어느 쪽이 "내용 보존 + 탐지기 분리"를 더 잘 만드는지가
파일럿의 핵심 질문이다.

  P1 = MASH 원논문 Stage 1 방식의 한국어 이식.
       원문(부록 B)은 "AI처럼 써라"가 아니라 일반 교정 지시다:
         "Polish the following English text for grammar, clarity, and flow."
       약한 윤문만으로 탐지기 라벨이 뒤집힌다는 것이 논문의 가정(영어 flip율 ≈95%).
  P2 = 팀 기존 프롬프트 v3 (mash/kci_humanization_dataset_v0/prompt_v3.txt).
       문장 단위 충실 재서술 — 개입 강도가 P1보다 훨씬 세다.

주의: 재서술은 인간 원문이 프롬프트에 통째로 들어가므로, 폐기된 조건 B와 같은
복사 유출 위험군이다. 생성 후 반드시 stage0_check_pairs.py 로 검사할 것.
"""
import argparse
import json
from pathlib import Path

# --- P1: MASH 부록 B Stage 1 프롬프트의 한국어 이식 ---------------------------
# 영어 원문: system "You are a professional English editor focusing on clarity and
# conciseness." / user "Polish the following English text for grammar, clarity, and
# flow. Output only the polished text — no introductory phrases, no explanations,
# no headings, and no markdown or asterisk * lists. Preserve any in-text citations
# exactly as they appear:\n\n {human_text}"
# 한국어 이식 시 변경한 곳: ① 언어 지정, ② 마크다운 금지 목록에 한국어 모델이 즐겨 쓰는
# 불릿("-", "•", 번호 목록) 추가, ③ 인용 표기 보존을 한국어 관례로 확장.
P1_SYSTEM = "당신은 명료성과 간결성에 집중하는 전문 국어 교정자입니다."
P1_USER = (
    "다음 한국어 글의 문법, 명료성, 흐름을 다듬으세요. "
    "다듬은 글만 출력하세요 — 도입 문구, 설명, 제목을 붙이지 말고, "
    "굵은 글씨·별표(*)·불릿(-, •)·번호 목록 같은 서식도 쓰지 마세요. "
    "인용 표기와 참고문헌 표기는 원문에 나온 그대로 유지하세요:\n\n{text}"
)

# --- P2: 팀 프롬프트 v3 (원문 그대로, {human_text} 자리만 치환) ----------------
P2_PATH_DEFAULT = "/workspace/mash/kci_humanization_dataset_v0/prompt_v3.txt"

# --- P3 계열: 파일럿 1차 결과에서 도출한 개선안 --------------------------------
# 관찰 (notes/43):
#   · P1(polish)  flip 77% — 잘 뒤집히지만 요약해버린다(수치 20% 유실, 22%가 길이 80% 미만)
#   · P2(충실 재서술) flip 37% — 보존은 되지만 베낀다(n그램 49%, 최대 217자 연속 복사)
#   · 두 변형 모두 **덜 베낀 글일수록 flip된다** (P1 성공군 n그램 25% vs 실패군 40%)
# 설계 원칙: P1의 재서술 자유도를 유지하되, 잃어버리는 것(수치·고유명사·분량)만 못 박는다.
# "요약하지 말라"는 P2에도 있었지만 P2는 동시에 "문장 단위 대응"을 요구해 복사를 유도했다.
# P3 는 문장 단위 대응을 요구하지 않는다 — 이것이 P2 와의 결정적 차이다.

P3A_SYSTEM = "당신은 명료성과 간결성에 집중하는 전문 국어 교정자입니다."
P3A_USER = (
    "다음 한국어 글의 문법, 명료성, 흐름을 다듬으세요. "
    "표현과 문장 구조는 자유롭게 바꾸되, 다음은 반드시 지키세요.\n"
    "- 원문에 나온 모든 숫자, 연도, 비율, 단위, 통계값을 그대로 유지할 것\n"
    "- 고유명사와 전문 용어를 그대로 유지할 것\n"
    "- 증가/감소, 유의/비유의 같은 방향과 결론을 바꾸지 말 것\n"
    "- 내용을 요약하거나 생략하지 말고, 분량을 원문의 90~110%로 유지할 것\n"
    "다듬은 글만 출력하세요 — 도입 문구, 설명, 제목을 붙이지 말고, "
    "굵은 글씨·별표(*)·불릿(-, •)·번호 목록 같은 서식도 쓰지 마세요.\n\n{text}"
)

# P3B: 문체 전이를 명시적으로 지시한다(MASH 가 하지 않은 방향).
# 복사를 가장 강하게 억제하지만 내용 보존이 깨질 위험도 가장 크다 — 실측으로 판단한다.
P3B_SYSTEM = "당신은 학술 문장을 다듬는 전문 국어 편집자입니다."
P3B_USER = (
    "다음 한국어 글을 같은 내용, 다른 문장으로 다시 쓰세요.\n"
    "- 원문의 문장을 그대로 옮기지 말고, 어휘와 문장 구조를 새로 짜세요. "
    "문장을 합치거나 나누어도 좋습니다.\n"
    "- 사실은 하나도 바꾸지 마세요: 숫자·연도·비율·단위·통계값, 고유명사와 전문 용어, "
    "증가/감소와 유의/비유의 같은 방향, 연구의 목적·방법·결과·결론을 모두 보존하세요.\n"
    "- 요약하지 말고 분량을 원문의 90~110%로 유지하세요.\n"
    "다시 쓴 글만 출력하세요 — 설명이나 제목, 마크다운 서식은 쓰지 마세요.\n\n{text}"
)

# P3C: 독자 대상을 바꿔 문체 변화를 유도한다(지시가 아니라 상황으로 유도).
P3C_SYSTEM = "당신은 학술지 편집위원으로서 투고 원고의 초록을 다듬습니다."
P3C_USER = (
    "다음 초록을 학술지 게재 기준에 맞게 다듬어 주세요. "
    "읽기 쉽고 자연스러운 문장이 되도록 표현을 고치되, "
    "연구의 사실관계는 조금도 바꾸지 마세요 — 숫자·연도·비율·단위, 고유명사와 전문 용어, "
    "증가/감소와 유의성의 방향, 목적·방법·결과·결론이 모두 원문과 같아야 합니다. "
    "내용을 줄이지 말고 분량은 원문의 90~110%로 유지하세요.\n"
    "다듬은 초록만 출력하세요 — 다른 말이나 서식은 붙이지 마세요.\n\n{text}"
)

VARIANTS = ("P1", "P2")
P3_VARIANTS = {"P3a": (P3A_SYSTEM, P3A_USER), "P3b": (P3B_SYSTEM, P3B_USER), "P3c": (P3C_SYSTEM, P3C_USER)}


def build(seed, variant, p2_template):
    text = seed["human_text"]
    if variant == "P1":
        return P1_SYSTEM, P1_USER.format(text=text)
    if variant == "P2":
        # prompt_v3.txt 는 말미에 "[원문]\n{human_text}" 형태를 갖는다
        return "", p2_template.replace("{human_text}", text)
    if variant in P3_VARIANTS:
        system, user = P3_VARIANTS[variant]
        return system, user.format(text=text)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="stage0_sample_pilot.py 산출물")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--p2-template", default=P2_PATH_DEFAULT)
    a = ap.parse_args()

    p2_template = ""
    if "P2" in a.variants:
        p2_template = Path(a.p2_template).read_text(encoding="utf-8")
        assert "{human_text}" in p2_template, "prompt_v3.txt 에 {human_text} 자리가 없다"

    seeds = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for s in seeds:
            for v in a.variants:
                system, prompt = build(s, v, p2_template)
                f.write(
                    json.dumps(
                        {
                            "doc_id": s["pair_id"],
                            "cond": v,  # generate.py 의 집계 축 = 프롬프트 변형
                            "domain": s["domain"],
                            # 재서술이므로 목표 길이 = 원문 길이 (max_tokens 여유는 --token-ratio 로)
                            "target_char": s["human_chars"],
                            "n_char": s["human_chars"],
                            "system": system,
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1

    print(f"인간 원문 {len(seeds)}편 × 변형 {len(a.variants)}종 = {n}건 → {out}")


if __name__ == "__main__":
    main()
