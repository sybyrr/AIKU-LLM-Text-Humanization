#!/usr/bin/env python3
"""반복 지표 3층 — 급성붕괴 / 문장 재사용 / 근사중복.

GRUEN(Zhu & Bhat 2020) non-redundancy 를 한국어로 이식하되, **두 항을 분리해서 낸다.**
2026-08-31 앵커 런(essay·persona 각 4모집단 600편)에서 두 항이 서로 반대 신호였기 때문이다.

  ① 급성붕괴  한 어절 30회 초과 반복      이진 %   — 게이트. 발생 샘플은 평가 무효
  ② 문장 재사용  동일하거나 포함되는 문장 쌍  연속값   — **품질 열화 지표**
  ③ 근사중복    GRUEN 나머지 4기준          연속값   — **인간다움 지표(품질 아님)**

왜 나누나: essay 실측에서 ③은 x_ai 7.72 < 인간 13.88 이라 humanizer 가 이 값을 올리는 것이
인간 수준 복원(성공)이었고, ②는 인간 0.16 인데 SFT 0.87 로 인간에게 없는 현상이 새로 생겼다.
둘을 합친 총 flag 로는 이 구분이 사라져 "반복이 늘었다 = 열화" 로 오독된다.

GRUEN 원본 규칙(main.py if_two_sentence_redundant) 그대로:
  a==b 또는 포함 → ②. 아니면 max(어절수)>=5 인 쌍만 아래 4개를 각각 +1 하여 ③.
    최장공통부분문자열 문자길이 > 0.8*min(len) · 그 어절수 > 0.8*min(어절수)
    편집거리 < 0.6*max(len) · 공통어절수 > 0.8*min(어절수)
  ※ >=5어절 가드가 대화의 짧은 맞장구가 반복으로 잡히는 것을 막는다.

문장 수의 제곱에 비례하므로 **쌍 1000개당**으로 정규화해 보고한다. 편집거리 항의 기준선은
도메인마다 달라(대화 0.92 / 에세이 13.88) 절대값 비교는 무의미하다 —
반드시 **같은 도메인 인간 원문 대비 배수**로 읽는다.

사용:
  python metrics_redundancy.py --in logs/anchor_essay.jsonl --group dec
  python metrics_redundancy.py --in logs/anchor_persona.jsonl --group dec --dialogue
"""
import argparse, collections, difflib, json, re, statistics as st

SENT = re.compile(r"(?<=[.!?])\s+")
MARK = re.compile(r"^\s*[AB]\s*[::]\s*")
MIN_WORDS = 5          # GRUEN 의 짧은 문장 가드
ACUTE_NGRAM = 6        # 급성붕괴: 문자 n-gram 길이
ACUTE_REPEAT = 20      # 급성붕괴: 같은 n-gram 반복 횟수 문턱


_KIWI = None


def _kiwi():
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return _KIWI


def sentences_regex(text, dialogue=False):
    """구 구현 — 종결부호 기준. 비교·이력 확인용으로만 남긴다.

    **인간 글에만 편향된다.** 마침표를 생략하는 글은 자를 근거가 없어 통째로 한 문장이 된다
    (실측: essay 인간 문서당 경계 1.81개 누락, AI 는 −0.07). 정답 검증에서 AI 생성문의
    종결부호를 지우면 F1 0.000 — 완전히 작동하지 않는다.
    """
    if dialogue:
        return [MARK.sub("", s).strip() for s in (text or "").split("\n") if s.strip()]
    return [s.strip() for s in SENT.split(text or "") if s.strip()]


def sentences(text, dialogue=False):
    """문장 분할 — kiwi.

    정답 검증(essay x_ai 150편, 종결부호를 지운 뒤 원래 경계를 되찾는지):
      정규식 F1 **0.000** · kiwi F1 **0.948**(정밀도 0.995) · 부호 있는 원문이면 kiwi 0.988.
    kiwi 는 어미·품사로 판단하므로 부호가 없어도 찾고, 없는 경계를 만들지 않는다.

    **대화는 발화(줄)로 먼저 자르고 그 안에서만 kiwi 를 적용한다.** 전체 텍스트에 걸면
    발화 경계를 넘나들며 재분절해 문장의 46%가 바뀌고, 순서만 섞은 널 변환이 dz +0.72
    (실제 효과의 2배)를 만든다. 줄 단위로 먼저 자르면 셔플 불변 검산을 통과한다(보존율 1.000).
    """
    k = _kiwi()
    if dialogue:
        out = []
        for ln in (text or "").split("\n"):
            ln = MARK.sub("", ln).strip()
            if ln:
                out += [x.text.strip() for x in k.split_into_sents(ln) if x.text.strip()]
        return out
    return [x.text.strip() for x in k.split_into_sents(text or "") if x.text.strip()]


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pair_scores(a, b):
    """(동일·포함 1/0, 근사중복 0~4). GRUEN 원본 기준."""
    if a == b or (a in b) or (b in a):
        return 1, 0
    A, B = a.split(), b.split()
    if max(len(A), len(B)) < MIN_WORDS:
        return 0, 0
    approx = 0
    m = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    if m.size > 0.8 * min(len(a), len(b)):
        approx += 1
    if len(a[m.a:m.a + m.size].strip().split()) > 0.8 * min(len(A), len(B)):
        approx += 1
    # 길이 차가 이미 임계를 넘으면 편집거리는 볼 것도 없다(비용 절감).
    if abs(len(a) - len(b)) < 0.6 * max(len(a), len(b)):
        if levenshtein(a, b) < 0.6 * max(len(a), len(b)):
            approx += 1
    if len([x for x in A if x in B]) > 0.8 * min(len(A), len(B)):
        approx += 1
    return 0, approx


def acute_collapse(text, n=ACUTE_NGRAM, k=ACUTE_REPEAT):
    """① 급성붕괴 — 공백을 지운 글자열에서 같은 n글자가 k회를 넘게 반복.

    **어절 반복(구 구현)은 공백에 의존해 가장 심한 붕괴를 놓쳤다.** 붕괴한 모델은
    구분자를 `·` `∼` 로 쓰기도 하는데, 그러면 `주권·주권·주권·…` 가 어절 하나가 되어
    반복 횟수가 1 로 계산된다(실측: 1,440자 글이 어절 6개). 신문 vanilla DPO 에서
    어절 기준 76.7% vs 문자 기준 96.7% — 20%p 를 놓쳤고, 젬마 LLM 이진판정(100%)과도
    문자 기준이 더 가깝다. 정상 집단(human·x_ai·SFT)은 두 기준 모두 0% 로 일치한다.

    어절 반복은 이 기준에 포함된다 — 같은 어절이 붙어 반복되면 같은 n-gram 도 반복된다.
    """
    s = re.sub(r"\s+", "", text or "")
    if len(s) < n:
        return False
    g = collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))
    return max(g.values()) > k


def acute_collapse_word(text, k=30):
    """구 구현(어절 반복) — 비교·이력 확인용으로만 남긴다."""
    toks = (text or "").split()
    return bool(toks) and max(collections.Counter(toks).values()) > k


def redundancy(text, dialogue=False):
    """②③ 을 쌍 1000개당으로 환산해 돌려준다."""
    s = sentences(text, dialogue)
    pairs = len(s) * (len(s) - 1) // 2
    ident = approx = 0
    for i in range(len(s) - 1):
        for j in range(i + 1, len(s)):
            a, b = pair_scores(s[i], s[j])
            ident += a
            approx += b
    scale = 1000 / pairs if pairs else 0.0
    return {"n_sent": len(s), "n_pair": pairs,
            "reuse": ident * scale,          # ② 문장 재사용
            "approx": approx * scale,        # ③ 근사중복
            "has_reuse": ident > 0,
            "acute": acute_collapse(text)}


def report(groups, order=None, baseline="human"):
    """모집단별 3층 표. 인간 기준선이 있으면 배수를 같이 낸다."""
    keys = order or sorted(groups)
    base = groups.get(baseline)
    print(f"{'모집단':<10}{'n':>5}{'급성붕괴':>9}{'문장재사용':>12}{'근사중복':>11}"
          f"{'재사용문서':>11}   인간대비(재사용/근사)")
    for k in keys:
        g = groups.get(k)
        if not g:
            continue
        reuse, approx = st.mean(x["reuse"] for x in g), st.mean(x["approx"] for x in g)
        rel = ""
        if base and k != baseline:
            br, ba = st.mean(x["reuse"] for x in base), st.mean(x["approx"] for x in base)
            rel = f"   {reuse/br:.1f}배 / {approx/ba:.1f}배" if br and ba else ""
        print(f"{k:<10}{len(g):>5}{100*sum(x['acute'] for x in g)/len(g):>8.0f}%"
              f"{reuse:>12.2f}{approx:>11.2f}"
              f"{100*sum(x['has_reuse'] for x in g)/len(g):>10.0f}%{rel}")
    print("  ② 문장재사용 = 품질 열화 · ③ 근사중복 = 인간다움(낮을수록 AI스러움). 쌍 1000개당.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--group", default="dec", help="모집단 필드명 (dec/cond/model)")
    ap.add_argument("--dialogue", action="store_true", help="대화 도메인: 발화(줄) 단위로 분할")
    ap.add_argument("--order", nargs="*", default=["human", "xai", "sft", "dpo"])
    ap.add_argument("--baseline", default="human")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ngram", type=int, default=ACUTE_NGRAM)
    ap.add_argument("--repeat", type=int, default=ACUTE_REPEAT)
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        if r.get("text"):
            d = redundancy(r["text"], args.dialogue)
            d["acute"] = acute_collapse(r["text"], args.ngram, args.repeat)
            groups[r.get(args.group, "?")].append(d)
    order = [k for k in args.order if k in groups] + \
            [k for k in sorted(groups) if k not in args.order]
    print(f"{args.inp} · {sum(len(v) for v in groups.values())}편"
          f" · {'발화' if args.dialogue else '문장'} 단위")
    report(groups, order, args.baseline)


if __name__ == "__main__":
    main()
