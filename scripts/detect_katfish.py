#!/usr/bin/env python3
"""KatFishNet 계열 특징 추출 + 로지스틱 회귀 탐지기.

원논문: Park et al., "KatFishNet: Detecting LLM-Generated Korean Text through
Linguistic Feature Analysis" (ACL 2025 Main). 한국어 세 축을 본다.

  1. 띄어쓰기   한국어는 띄어쓰기 규칙이 느슨해서 사람 글에 흔들림이 있다.
                의존명사·보조용언 앞 띄어쓰기, 어절 길이 분포로 잡는다.
  2. POS n-gram 다양성   품사 배열의 반복성. n=1..5 로 문장별 유형/토큰 비.
  3. 쉼표       한국어는 영어보다 쉼표를 덜 쓴다. 빈도·위치·앞뒤 품사를 본다.

원 저장소(github.com/Shinwoo-Park/katfishnet)와 다른 점 — **재구현이다.**
  * 저장소는 전처리된 특징 pkl 만 배포하고 임의 텍스트용 파이프라인이 없다.
    쉼표/POS n-gram 함수는 lingustic_analysis/ 에 있으나 띄어쓰기 쪽은 없다.
  * 원본은 Kkma(konlpy) + kss 를 쓰는데 이 환경엔 JVM 이 없다. **Kiwi 로 대체**했다.
    태그셋이 달라 특징 값이 원논문과 같지 않다. 분류기를 우리 데이터로 새로 학습하므로
    자기일관성은 유지되지만, **원논문 수치와 직접 비교하면 안 된다.**
  * 원본 KatFish 는 논술·시·논문초록이고 우리는 신문이다. 도메인이 다르다.

사용:
    # 특징만 뽑아 조건별 비교
    python detect_katfish.py stats --human ../dataset/human/mash_pilot_10.jsonl \
        --gen ../dataset/generations/mash_pilot_generations.jsonl ../dataset/generations/mash_ablation.jsonl
"""
import argparse, json, re, statistics as st
from collections import Counter

from kiwipiepy import Kiwi

# 의존명사·보조용언은 앞말과 띄어 쓰는 게 원칙이지만 실제 글에서는 자주 붙는다.
# LLM 은 규범을 더 잘 지킨다는 게 원논문의 관찰이다.
BOUND = {"NNB", "NR"}          # 의존명사, 수사
AUX = {"VX"}                   # 보조용언
COMMA = ","

_kiwi = None


def kiwi():
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def sentences(text):
    return [s.text for s in kiwi().split_into_sents(text) if s.text.strip()]


def features(text):
    """글 1건 → 특징 벡터(dict). 세 축을 한 번의 형태소 분석으로 뽑는다."""
    k = kiwi()
    sents = sentences(text)
    if not sents:
        return None

    # ── 1. 띄어쓰기 ────────────────────────────────────────────
    # 의존명사/보조용언이 앞말과 붙어 있는지. 원문 표면형에서 직접 센다.
    bound_total = bound_attached = 0
    aux_total = aux_attached = 0
    for tok in k.tokenize(text):
        if tok.tag in BOUND or tok.tag in AUX:
            attached = tok.start > 0 and text[tok.start - 1] not in " \n\t"
            if tok.tag in BOUND:
                bound_total += 1
                bound_attached += attached
            else:
                aux_total += 1
                aux_attached += attached
    eojeol = [w for w in re.split(r"\s+", text) if w]
    elen = [len(w) for w in eojeol]

    # ── 2. POS n-gram 다양성 ──────────────────────────────────
    pos_per_sent = [[t.tag for t in k.tokenize(s)] for s in sents]
    div = {}
    for n in (1, 2, 3, 4, 5):
        vals = []
        for p in pos_per_sent:
            grams = [tuple(p[i - n + 1:i + 1]) for i in range(n - 1, len(p))]
            if grams:
                vals.append(len(set(grams)) / len(grams))
        div[f"pos{n}_div"] = st.mean(vals) if vals else 0.0

    # ── 3. 쉼표 ───────────────────────────────────────────────
    n_comma = text.count(COMMA)
    with_comma = sum(1 for s in sents if COMMA in s)
    rel_pos, seg_len, around = [], [], []
    for s in sents:
        toks = k.tokenize(s)
        idx = [i for i, t in enumerate(toks) if t.form == COMMA]
        for i in idx:
            rel_pos.append(i / max(1, len(toks) - 1))
            if i > 0:
                around.append((toks[i - 1].tag, toks[i + 1].tag if i + 1 < len(toks) else "END"))
        prev = 0
        for pos in [t.start for t in toks if t.form == COMMA]:
            seg_len.append(pos - prev)
            prev = pos

    return {
        # 띄어쓰기
        "bound_attach": bound_attached / bound_total if bound_total else 0.0,
        "aux_attach": aux_attached / aux_total if aux_total else 0.0,
        "eojeol_len_mean": st.mean(elen) if elen else 0.0,
        "eojeol_len_std": st.pstdev(elen) if len(elen) > 1 else 0.0,
        # POS 다양성
        **div,
        # 쉼표
        "comma_per_sent": n_comma / len(sents),
        "comma_sent_ratio": with_comma / len(sents),
        "comma_relpos_mean": st.mean(rel_pos) if rel_pos else 0.0,
        "comma_relpos_std": st.pstdev(rel_pos) if len(rel_pos) > 1 else 0.0,
        "comma_seg_mean": st.mean(seg_len) if seg_len else 0.0,
        "comma_around_div": (len(set(around)) / len(around)) if around else 0.0,
        # 참고
        "sent_len_mean": st.mean([len(s) for s in sents]),
        "sent_len_std": st.pstdev([len(s) for s in sents]) if len(sents) > 1 else 0.0,
    }


KEYS = None


def load(human_path, gen_paths):
    items = []
    for l in open(human_path):
        r = json.loads(l)
        items.append(("human", "human", r["doc_id"], r["body"]))
    for p in gen_paths:
        for l in open(p):
            r = json.loads(l)
            if r.get("text"):
                items.append((r["cond"], r["model"], r["doc_id"], r["text"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats"])
    ap.add_argument("--human", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load(args.human, args.gen)
    print(f"{len(items)}건 특징 추출 중...", flush=True)
    rows = []
    for cond, model, doc_id, text in items:
        f = features(text)
        if f:
            rows.append({"cond": cond, "model": model, "doc_id": doc_id, **f})

    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    keys = [k for k in rows[0] if k not in ("cond", "model", "doc_id")]
    conds = ["human"] + [c for c in dict.fromkeys(r["cond"] for r in rows) if c != "human"]
    hum = [r for r in rows if r["cond"] == "human"]
    print(f"\n{'특징':<20}" + "".join(f"{c:>12}" for c in conds))
    print("-" * (20 + 12 * len(conds)))
    for k in keys:
        line = f"{k:<20}"
        for c in conds:
            v = [r[k] for r in rows if r["cond"] == c]
            line += f"{st.mean(v):>12.3f}"
        print(line)
    print(f"\n{'':<20}" + "".join(f"{len([r for r in rows if r['cond']==c]):>12}" for c in conds))


if __name__ == "__main__":
    main()
