#!/usr/bin/env python3
"""평가셋용 — OA학술지 논문의 **본문 + 한글초록**을 짝지어 수집한다.

왜 본문이 필요한가:
  평가용 AI 초록을 "인간 초록의 재서술"로 만들면 학습쌍 $x_{ai}$ 와 구성 방식이 같아져,
  humanizer 가 학습한 바로 그 분포를 평가하게 된다(ASR 과대평가).
  **본문을 주고 초록을 쓰게 하면** 과제 자체가 달라져 이 문제가 사라지고,
  연구자가 실제로 LLM 을 쓰는 방식과도 가장 가깝다.

두 오퍼레이션을 조인한다 (둘 다 OA학술지 한정):
  · openApiD267List  OA학술지-초록 관리 — USELANG=kor 인 PARA 를 ABSSEQ 순으로 이어붙인다
  · openApiD268List  OA학술지-본문 관리 — TYPE=B101(본문 문단)만 BODYSEQ 순으로 이어붙인다
    (B2xx/B3xx 는 장·절 제목이라 제외. HTML 엔티티가 이중 이스케이프돼 있어 두 번 푼다.)

본문 발췌:
  본문 중앙값이 2만 자를 넘어 슬롯 컨텍스트(4096토큰 ≈ 6~8천 자)에 안 들어간다.
  앞부분만 자르면 서론만 남아 결과·결론이 빠지므로, **앞 절반 + 뒤 절반**을 떼어 붙인다.
  초록이 담아야 할 목적(앞)과 결과·결론(뒤)이 둘 다 들어가게 하려는 것이다.

사용:
  python3 collect_kci_bodies.py --out mash/kci_eval_bodies.jsonl --target 300
"""
import argparse
import collections
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://apis.data.go.kr/B552540/KCIOpenApi/artiInfo"
KEY_FILE = "/workspace/.API_KEY"
HANGUL = re.compile(r"[가-힣]")


def unesc(s):
    # API 응답의 엔티티가 이중 이스케이프돼 있다 (&amp;#8544; → &#8544; → Ⅰ)
    return html.unescape(html.unescape(s or ""))


def fetch(op, key, page, cnt, retries=4):
    q = urllib.parse.urlencode({"serviceKey": key, "pageNo": page, "recordCnt": cnt})
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{op}?{q}", timeout=90) as r:
                root = ET.fromstring(r.read().decode("utf-8", "replace"))
            if root.findtext(".//resultCode") != "00":
                raise RuntimeError(root.findtext(".//resultMsg"))
            return [{c.tag: unesc(c.text) for c in it} for it in root.iter("item")]
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return []


def harvest(op, key, pages, cnt, keep, label):
    """pages 만큼 훑으며 keep(item) 이 참인 레코드를 ARTIID 별로 모은다."""
    out = collections.defaultdict(list)
    for p in range(1, pages + 1):
        try:
            items = fetch(op, key, p, cnt)
        except Exception as e:
            print(f"  {label} p{p} 실패: {e}", file=sys.stderr)
            continue
        if not items:
            break
        for it in items:
            if keep(it):
                out[it.get("ARTIID", "")].append(it)
        if p % 20 == 0:
            print(f"  {label}: {p}페이지 · 논문 {len(out)}편", flush=True)
    return out


def joined(parts, seq_key):
    return "\n".join(p.get("PARA", "") for p in sorted(parts, key=lambda x: x.get(seq_key, "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--abst-pages", type=int, default=12)
    ap.add_argument("--body-pages", type=int, default=120)
    ap.add_argument("--min-abst", type=int, default=300)
    ap.add_argument("--max-abst", type=int, default=1200)
    ap.add_argument("--min-body", type=int, default=3000)
    ap.add_argument("--excerpt", type=int, default=5000, help="본문 발췌 총 길이(앞+뒤)")
    a = ap.parse_args()

    key = Path(KEY_FILE).read_text().strip()

    print("한글초록 수집…", flush=True)
    absts = harvest("openApiD267List", key, a.abst_pages, 1000,
                    lambda it: it.get("USELANG") == "kor", "초록")
    print(f"  → 초록 보유 논문 {len(absts):,}편", flush=True)

    print("본문 수집…", flush=True)
    bodies = harvest("openApiD268List", key, a.body_pages, 1000,
                     lambda it: it.get("TYPE") == "B101", "본문")
    print(f"  → 본문 보유 논문 {len(bodies):,}편", flush=True)

    rows, stats = [], collections.Counter()
    for aid in sorted(set(absts) & set(bodies)):
        stats["교집합"] += 1
        abst = joined(absts[aid], "ABSSEQ").strip()
        body = joined(bodies[aid], "BODYSEQ").strip()
        if not (a.min_abst <= len(abst) <= a.max_abst):
            stats["초록 길이 밖"] += 1
            continue
        if len(body) < a.min_body:
            stats["본문 짧음"] += 1
            continue
        b = re.sub(r"\s", "", abst)
        hr = len(HANGUL.findall(abst)) / len(b) if b else 0
        if hr < 0.30:
            stats["한글비 미달"] += 1
            continue
        # 앞 절반 + 뒤 절반 발췌 — 목적(앞)과 결과·결론(뒤)을 모두 담는다
        h = a.excerpt // 2
        excerpt = body if len(body) <= a.excerpt else body[:h] + "\n(…중략…)\n" + body[-h:]
        rows.append({
            "kci_article_id": aid,
            "original_abstract": abst,
            "body_excerpt": excerpt,
            "body_chars": len(body),
            "abstract_chars": len(abst),
            "abstract_hangul_ratio": round(hr, 4),
            "source": "data.go.kr/15085348/openApiD267List+D268List",
        })
        stats["채택"] += 1
        if len(rows) >= a.target:
            break

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n" + " · ".join(f"{k} {v:,}" for k, v in stats.most_common()))
    print(f"{len(rows):,}편 → {out}")
    if rows:
        bl = sorted(r["body_chars"] for r in rows)
        al = sorted(r["abstract_chars"] for r in rows)
        print(f"  본문 중앙값 {bl[len(bl)//2]:,}자 (발췌 후 최대 {a.excerpt:,}자)"
              f" · 초록 중앙값 {al[len(al)//2]}자")


if __name__ == "__main__":
    main()
