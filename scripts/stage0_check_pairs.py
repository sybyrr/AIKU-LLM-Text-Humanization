#!/usr/bin/env python3
"""Stage 0 파일럿 — (인간 원문, AI 재서술문) 짝의 복사 유출 · 내용 보존 검사.

check_copying.py 를 대체한다. 그쪽은 "인간 도입부 2문장"과만 비교하도록 되어 있어
재서술 트랙에 맞지 않고, 순수 파이썬 LCS 라 본문 전체 비교에서 O(n·m) 으로 터진다.

검사 두 갈래:

  ① 복사 유출 — 재서술은 인간 원문이 프롬프트에 통째로 들어가므로, 폐기된 조건 B와
     같은 위험군이다. 지표와 기준선은 기존 트랙에서 그대로 승계한다:
       · 12자 n그램 겹침률 — 정상 ≈1%, 조건 B(유출) 20.4%
       · 최장 공통 부분문자열(LCS) — 정상 9~10자, 조건 B 28.2자
     판정선: LCS > 25자 또는 n그램 > 10% → 유출 의심 (summarize.py 의 --max-copy 25 와 일치)

  ② 내용 보존 — MASH 는 이 검사를 하지 않았고(BERTScore 사후 보고뿐), 그래서 우리가
     새로 만든다. 자동으로 잡을 수 있는 것만 잡는다. 조사로 인한 주객 전도처럼
     자동 검출이 어려운 항목은 사람 검수 표본으로 넘긴다(handoff.md §8).
       · 숫자·연도·비율·단위 보존과 환각
       · 방향어 보존 (증가/감소, 유의/비유의, 정/부 …)
       · 부정 표현 개수 변화
       · 고유명사(NNP) 보존율 — kiwipiepy 있을 때만
       · 길이 비율 (원문의 80~120%)

사용:
  python3 stage0_check_pairs.py --human pilot/data/seed_abstract.jsonl \
      --gen pilot/gen/qwen3-8b.jsonl --out pilot/qa/qwen3-8b.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

NGRAM = 12  # check_copying.py 와 동일 (기준선 수치를 그대로 쓰기 위해)
LCS_LIMIT = 25
NGRAM_LIMIT = 10.0

NUM = re.compile(r"\d+(?:[.,]\d+)*\s*(?:%|퍼센트|명|건|개|년|월|일|시간|배|원|억|만|천)?")
MARKDOWN = re.compile(r"(\*\*|^#{1,4}\s|^\s*[-*•]\s|^\s*\d+\.\s)", re.M)
NEG = re.compile(r"(않|못하|못한|없|아니|불가|미흡|비유의)")

# 방향이 뒤집히면 결론이 바뀌는 표현들 (prompt_v3 가 명시적으로 금지한 항목)
DIRECTION = {
    "증가": r"증가|늘어|상승|높아|많아",
    "감소": r"감소|줄어|하락|낮아|적어",
    "유의": r"유의(?!하지|하지 않)",
    "비유의": r"유의하지 않|유의미하지 않|비유의",
    "정적": r"정\(\+\)|정적 (?:상관|영향)|양(?:\(\+\)|의 상관)",
    "부적": r"부\(-\)|부적 (?:상관|영향)|음(?:\(-\)|의 상관)",
}


def ngrams(s, n=NGRAM):
    s = re.sub(r"\s+", "", s)
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def overlap_pct(human, ai):
    """인간 원문의 12자 조각 중 AI 글에 그대로 남은 비율(%)."""
    h = ngrams(human)
    if not h:
        return 0.0
    a = ngrams(ai)
    return len(h & a) / len(h) * 100


def lcs_len(a, b):
    """최장 공통 부분문자열 길이. SequenceMatcher 는 C 가속이라 전체 비교에도 견딘다."""
    a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    if not a or not b:
        return 0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def nums(s):
    return Counter(re.sub(r"\s+", "", t) for t in NUM.findall(s))


def directions(s):
    return {k: len(re.findall(v, s)) for k, v in DIRECTION.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="stage0_sample_pilot.py 산출물")
    ap.add_argument("--gen", required=True, help="generate.py 산출물")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lcs-limit", type=int, default=LCS_LIMIT)
    ap.add_argument("--ngram-limit", type=float, default=NGRAM_LIMIT)
    ap.add_argument("--no-kiwi", action="store_true")
    a = ap.parse_args()

    kiwi = None
    if not a.no_kiwi:
        try:
            from kiwipiepy import Kiwi

            kiwi = Kiwi()
        except Exception as e:  # 없으면 고유명사 검사만 건너뛴다
            print(f"kiwipiepy 없음 — 고유명사 검사 생략 ({e})", file=sys.stderr)

    def propn(s):
        if not kiwi:
            return None
        return {t.form for t in kiwi.tokenize(s) if t.tag == "NNP"}

    humans = {}
    for l in open(a.human, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            humans[r["pair_id"]] = r

    rows, skipped = [], 0
    for l in open(a.gen, encoding="utf-8"):
        if not l.strip():
            continue
        g = json.loads(l)
        if not g.get("text"):  # 오류 레코드
            skipped += 1
            continue
        h = humans.get(g["doc_id"])
        if h is None:
            skipped += 1
            continue
        ht, at = h["human_text"], g["text"]

        hn, an = nums(ht), nums(at)
        missing = sum((hn - an).values())          # 원문에 있었는데 사라진 수치
        hallucinated = sum((an - hn).values())     # 원문에 없는데 새로 생긴 수치
        # 방향어는 "개수 변화"가 아니라 "범주 소멸"만 본다.
        # 재서술이면 표현이 바뀌는 것이 정상이라("증가"→"늘어남") 개수 비교는 오검출이 심하다.
        # 결론이 뒤집히는 경우는 원문에 있던 방향 범주가 통째로 사라질 때다.
        hd, ad = directions(ht), directions(at)
        dir_delta = {k: ad[k] - hd[k] for k in DIRECTION if hd[k] > 0 and ad[k] == 0}

        hp, apn = propn(ht), propn(at)
        propn_keep = None
        if hp is not None and hp:
            propn_keep = round(len(hp & apn) / len(hp) * 100, 1)

        lcs = lcs_len(ht, at)
        ov = round(overlap_pct(ht, at), 2)
        ratio = round(len(at) / len(ht) * 100, 1)

        leak = lcs > a.lcs_limit or ov > a.ngram_limit
        preserved = (
            missing == 0
            and hallucinated == 0
            and not dir_delta
            and 80 <= ratio <= 130
            and (propn_keep is None or propn_keep >= 80)
        )

        rows.append(
            {
                "pair_id": g["doc_id"],
                "variant": g["cond"],
                "model": g["model"],
                "domain": h.get("domain", ""),
                "human_chars": len(ht),
                "ai_chars": len(at),
                "len_ratio": ratio,
                "lcs": lcs,
                "ngram_overlap": ov,
                "leak_suspect": leak,
                "nums_missing": missing,
                "nums_hallucinated": hallucinated,
                "direction_delta": dir_delta,
                "neg_delta": len(NEG.findall(at)) - len(NEG.findall(ht)),
                "propn_keep_pct": propn_keep,
                "has_markdown": bool(MARKDOWN.search(at)),
                "preserved": preserved,
            }
        )

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 요약 — 변형(프롬프트 계열)별로 나눠 본다. 이게 파일럿의 판단 근거다.
    print(f"검사 {len(rows)}건 (제외 {skipped}건) → {a.out}\n")
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["variant"]), []).append(r)
    hdr = f"{'모델':<14}{'변형':<6}{'n':>4}{'길이%':>7}{'LCS':>6}{'n그램%':>8}{'유출':>6}{'보존':>7}{'MD':>5}"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(by):
        v = by[k]
        n = len(v)
        print(
            f"{k[0]:<14}{k[1]:<6}{n:>4}"
            f"{sum(r['len_ratio'] for r in v)/n:>7.0f}"
            f"{sum(r['lcs'] for r in v)/n:>6.1f}"
            f"{sum(r['ngram_overlap'] for r in v)/n:>8.1f}"
            f"{sum(r['leak_suspect'] for r in v):>6}"
            f"{sum(r['preserved'] for r in v)/n*100:>6.0f}%"
            f"{sum(r['has_markdown'] for r in v):>5}"
        )
    print("\n기준선(기존 트랙): 정상 LCS 9~10자 · n그램 ≈1% / 유출된 조건 B: LCS 28.2자 · n그램 20.4%")


if __name__ == "__main__":
    main()
