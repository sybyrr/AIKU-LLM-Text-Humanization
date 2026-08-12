#!/usr/bin/env python3
"""조건 E 용 명사 목록 추출 (Kiwi).

조건 B(도입부 2문장)를 대체한다. B 는 인간 문장을 그대로 주기 때문에 모델이 복제해
버렸다(Qwen3-14B 최악 132자 100% 복사). 명사 목록은 문장이 아니라서 복제 자체가
구조적으로 불가능하면서 내용은 전달된다.

선택 기준:
  · NNP(고유명사)만 뽑으면 7개뿐이고 '금융위원회' 같은 핵심 기관명이 NNG 로
    분류돼 빠진다. 그래서 NNP+NNG 를 함께 쓴다.
  · Kiwi 가 복합어를 쪼개므로(금융/위원회) 원문에서 붙어 있던 토큰은 다시 합친다.
  · 빈도순은 '김, 이, 때' 같은 껍데기가 올라온다. 한국어 기사는 리드에 육하원칙이
    몰려 있으므로 등장 순서가 핵심어를 앞에 놓는다.
  · 1글자 명사는 2회차 언급의 성씨 조각('김','이')이 대부분이라 버린다.
"""
import argparse, json
from kiwipiepy import Kiwi

def merge_nouns(kiwi, text):
    """인접 명사 토큰을 원문에서 붙어 있으면 복합명사로 복원"""
    out, cur, end = [], None, None
    for t in kiwi.tokenize(text):
        if t.tag in ("NNG", "NNP"):
            if cur is not None and t.start == end:
                cur += t.form
            else:
                if cur:
                    out.append(cur)
                cur = t.form
            end = t.start + t.len
        else:
            if cur:
                out.append(cur)
            cur, end = None, None
    if cur:
        out.append(cur)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/workspace/dataset/human/full_base.jsonl")
    ap.add_argument("--out", default="/workspace/dataset/prompts/full_nouns.jsonl")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-len", type=int, default=2)
    args = ap.parse_args()

    kiwi = Kiwi()
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    out = []
    for r in rows:
        ns = [w for w in merge_nouns(kiwi, r["body"]) if len(w) >= args.min_len]
        uniq = list(dict.fromkeys(ns))[:args.top]          # 등장 순서, 중복 제거
        s = ", ".join(uniq)
        out.append({"doc_id": r["doc_id"], "nouns": s,
                    "n_nouns": len(uniq), "noun_char": len(s)})
        print(f"[{len(uniq):2d}개 {len(s):3d}자] {s[:70]}")

    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\n{len(out)}건 | 평균 {sum(o['n_nouns'] for o in out)/len(out):.1f}개 "
          f"/ {sum(o['noun_char'] for o in out)/len(out):.0f}자")

if __name__ == "__main__":
    main()
