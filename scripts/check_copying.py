#!/usr/bin/env python3
"""조건 B 가 인간 도입부를 그대로 베꼈는지 측정.

조건 B 만 프롬프트에 인간 산문(도입부 2문장)이 들어간다. A·C 는 안 들어가므로
A·C 의 겹침이 자연 발생 수준(제목·고유명사·상투구 때문에 우연히 겹치는 정도)이고,
B 가 그보다 얼마나 초과하는지가 실제 베끼기 양이다.

지표 2개:
  LCS   생성문과 도입부의 최장 공통 부분문자열(글자). 통째로 옮겼는지 본다.
  N그램 도입부의 12자 조각 중 몇 %가 생성문에 그대로 있는지. 조각조각 옮겼는지 본다.
"""
import json, sys
from collections import defaultdict

N = 12

def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best = v
        prev = cur
    return best

def grams(s, n=N):
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

def sents2(body):
    """조건 B 프롬프트에 들어간 것과 같은 방식으로 앞 2문장 추출"""
    flat = body.replace("\n\n", " ")
    parts = flat.replace(". ", ".|").split("|")
    return " ".join(parts[:2])

base = {r["doc_id"]: r for r in map(json.loads, open("/workspace/dataset/human/full_base.jsonl"))}
gens = [json.loads(l) for l in open("/workspace/dataset/generations/full_generations.jsonl") if l.strip()]

rows = []
for g in gens:
    if not g.get("text"):        # 생성 실패 건(오류 기록만 있음)은 건너뛴다
        continue
    b = base[g["doc_id"]]
    lead = sents2(b["body"])
    text = g["text"]
    lg, tg = grams(lead), grams(text)
    rows.append({
        "cond": g["cond"], "model": g["model"], "doc_id": g["doc_id"],
        "lcs_lead": lcs(text, lead),
        "gram_lead": round(100 * len(lg & tg) / max(1, len(lg))),
        "lead_char": len(lead),
    })

agg = defaultdict(list)
for r in rows:
    agg[(r["cond"], r["model"])].append(r)

print(f"{'조건':<4} {'모델':<13} {'LCS 평균':>9} {'LCS 최대':>9} {'N그램 평균':>11} {'N그램 최대':>11}")
print("-" * 62)
for k in sorted(agg):
    v = agg[k]
    n = len(v)
    print(f"{k[0]:<4} {k[1]:<13} "
          f"{sum(x['lcs_lead'] for x in v)/n:>8.1f}자 {max(x['lcs_lead'] for x in v):>8}자 "
          f"{sum(x['gram_lead'] for x in v)/n:>10.1f}% {max(x['gram_lead'] for x in v):>10}%")

print()
for cond in "AEC":
    v = [r for r in rows if r["cond"] == cond]
    print(f"조건 {cond}: LCS 평균 {sum(x['lcs_lead'] for x in v)/len(v):.1f}자 · "
          f"N그램 평균 {sum(x['gram_lead'] for x in v)/len(v):.1f}%")

worst = sorted(rows, key=lambda r: -r["lcs_lead"])[:5]
print("\n겹침 상위 5건")
for r in worst:
    lead = sents2(base[r["doc_id"]]["body"])
    print(f"  [{r['cond']}/{r['model']}] LCS {r['lcs_lead']}자 (도입부 {r['lead_char']}자 중) "
          f"N그램 {r['gram_lead']}%")
