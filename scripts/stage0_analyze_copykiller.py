#!/usr/bin/env python3
"""카피킬러 결과확인서(PDF) 파싱 → 유효쌍 수율·용량반응 분석.

카피킬러는 공개 API가 없고 결과가 PDF 확인서로만 나온다. 그 표를 긁어
manifest.csv(정답표) 및 보존 검사 결과와 조인한다.

⚠️ PDF 표 파싱 주의: pdftotext -layout 출력에서 파일명이 긴 행은 번호/비율과
   **다른 줄로 밀린다**(350건 중 5건). 정규식 한 방으로는 조용히 누락되므로
   인접 줄에서 고아 파일명을 회수한다. 파싱 수가 manifest 수와 같은지 반드시 확인.

유효쌍의 정의 — 세 조건을 모두 만족해야 한다:
  ① 인간 원문이 인간으로 판정 (ck < τ)   ← 이게 깨지면 짝 자체가 무의미
  ② AI 재서술이 AI로 판정  (ck ≥ τ)      ← MASH가 말하는 label flip
  ③ 내용 보존 게이트 통과 (stage0_check_pairs.py)

사용:
  python3 stage0_analyze_copykiller.py \
      --pdf "카피킬러 ... batch2 ....pdf" --manifest pilot/export/batch2/manifest.csv \
      --pdf "...batch1...pdf"             --manifest pilot/export/batch1/manifest.csv \
      --qa pilot/qa/abstract_qwen3-8b.jsonl --qa pilot/qa/abstract_p3_qwen.jsonl \
      --out pilot/scores/copykiller_all.jsonl
"""
import argparse
import collections
import csv
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REC = re.compile(r"^\s*(\d+)\s+(\d{11})\s+(?:([A-Z]\d{4,7}\.\w+)\s+)?(\d+)%\s*$")
LONE = re.compile(r"^\s*([A-Z]\d{4,7}\.\w+)\s*$")


def parse_pdf(path):
    """결과확인서에서 {파일명: AI작성률} 을 뽑는다."""
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, check=True).stdout
    lines = txt.split("\n")
    out = {}
    for i, l in enumerate(lines):
        m = REC.match(l)
        if not m:
            continue
        fn = m.group(3)
        if not fn:  # 파일명이 인접 줄로 밀린 경우 회수
            for j in (i - 1, i + 1, i - 2, i + 2):
                if 0 <= j < len(lines):
                    lm = LONE.match(lines[j])
                    if lm and lm.group(1) not in out:
                        fn = lm.group(1)
                        break
        if fn:
            out[fn] = int(m.group(4))
    return out


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--qa", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--tau", type=int, default=50, help="AI 판정 임계값(%)")
    ap.add_argument("--model", default="qwen3-8b", help="용량반응 분석 대상 모델")
    a = ap.parse_args()
    if len(a.pdf) != len(a.manifest):
        sys.exit("--pdf 와 --manifest 는 같은 수로 짝지어 줘야 한다")

    recs = []
    for pdf, mani in zip(a.pdf, a.manifest):
        scores = parse_pdf(pdf)
        rows = list(csv.DictReader(open(mani, encoding="utf-8")))
        miss = [r["file"] for r in rows if r["file"] not in scores]
        print(f"{Path(pdf).name}: 파싱 {len(scores)} · manifest {len(rows)} · 누락 {len(miss)}")
        if miss:
            sys.exit(f"⚠ 파싱 누락 {len(miss)}건 {miss[:5]} — 표 형식 확인 필요")
        for r in rows:
            recs.append({**r, "ck_ai_pct": scores[r["file"]]})

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"→ {a.out} ({len(recs)}건)\n")

    TAU = a.tau
    hum, ai = {}, collections.defaultdict(dict)
    for r in recs:
        if r["label"] == "human":
            hum[r["pair_id"]] = r["ck_ai_pct"]
        else:
            ai[(r["model"], r["variant"])][r["pair_id"]] = r["ck_ai_pct"]

    pres = {}
    for f in a.qa:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                q = json.loads(l)
                pres[(q["model"], q["variant"], q["pair_id"])] = q["preserved"]
    ov = collections.defaultdict(list)
    for f in a.qa:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                q = json.loads(l)
                ov[(q["model"], q["variant"])].append(q["ngram_overlap"])

    # ── 인간 오탐률 ────────────────────────────────────────────────
    fp = sum(1 for v in hum.values() if v >= TAU)
    lo, hi = wilson(fp, len(hum))
    print(f"인간 오탐률: {fp}/{len(hum)} = {fp/len(hum)*100:.1f}%  (95% CI {lo:.1f}~{hi:.1f}%)\n")

    # ── 변형별 유효쌍 수율 ─────────────────────────────────────────
    print("=== 유효쌍 수율 (인간=인간 AND AI=AI AND 내용보존) ===")
    print(f"{'모델':<16}{'변형':<6}{'대상':>5}{'인간OK':>7}{'flip':>6}{'유효쌍':>7}{'수율':>7}")
    print("-" * 54)
    for k in sorted(ai):
        ids = [p for p in ai[k] if p in hum]
        hok = [p for p in ids if hum[p] < TAU]
        flip = [p for p in hok if ai[k][p] >= TAU]
        good = [p for p in flip if pres.get((k[0], k[1], p))]
        print(f"{k[0]:<16}{k[1]:<6}{len(ids):>5}{len(hok):>7}{len(flip):>6}"
              f"{len(good):>7}{len(good)/len(ids)*100:>6.0f}%")

    # ── 용량반응: 복사율 ↔ AI 판정률 ───────────────────────────────
    pts = [(sum(ov[k]) / len(ov[k]),
            sum(1 for v in ai[k].values() if v >= TAU) / len(ai[k]) * 100, k[1])
           for k in ai if k[0] == a.model and k in ov]
    if len(pts) >= 3:
        print(f"\n=== 용량반응 ({a.model}, 동일 원문) ===")
        print(f"{'변형':<6}{'12자겹침':>9}{'AI판정':>8}")
        print("-" * 24)
        for x, y, v in sorted(pts):
            print(f"{v:<6}{x:>8.1f}%{y:>7.0f}%")
        print(f"\n피어슨 r = {pearson([p[0] for p in pts], [p[1] for p in pts]):.3f} (n={len(pts)}점)")

    # ── arm 조합 커버리지 ─────────────────────────────────────────
    V = sorted(v for m, v in ai if m == a.model)
    if len(V) >= 2:
        base = [p for p in ai[(a.model, V[0])] if p in hum and hum[p] < TAU]
        ok = {v: {p for p in base
                  if ai[(a.model, v)].get(p, 0) >= TAU and pres.get((a.model, v, p))} for v in V}
        print(f"\n=== arm 조합 (인간측 통과 {len(base)}편 기준) ===")
        print(f"{'조합':<18}{'커버 원문':>9}{'총 유효쌍':>10}{'원문당':>8}")
        print("-" * 46)
        for k in range(1, min(4, len(V)) + 1):
            best = max(itertools.combinations(V, k),
                       key=lambda c: (len(set().union(*[ok[v] for v in c])),
                                      sum(len(ok[v]) for v in c)))
            u = set().union(*[ok[v] for v in best])
            tot = sum(len(ok[v]) for v in best)
            print(f"{'+'.join(best):<18}{len(u):>6} ({len(u)/len(base)*100:>3.0f}%)"
                  f"{tot:>10}{tot/len(base):>8.2f}")


if __name__ == "__main__":
    main()
