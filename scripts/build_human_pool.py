#!/usr/bin/env python3
"""인간 초록 풀 통합 — 인수받은 v0(1,497편)와 새로 수집한 v1을 하나로 합친다.

두 파일은 출처가 달라 필드명이 어긋난다:
  · v0 (`kci_human_ko_cap100.jsonl`, 김용재님 인수분) — research_field 가 한글 분야명
  · v1 (`collect_kci.py` 산출)                       — research_field_code 가 코드(B210000)
아래에서 `research_field` 한 이름으로 맞춘다. 코드는 `FIELD:` 접두를 붙여 출처를 구분한다
(층화 표집에서 둘이 섞여도 서로 다른 층으로 잡히게 하려는 것 — 이름과 코드는 매핑이 없다).

kci_article_id 기준으로 중복을 제거한다. collect_kci.py 가 이미 --exclude 로 걸렀지만
재실행·부분 수집이 섞일 수 있어 여기서 한 번 더 본다.

사용:
  python3 build_human_pool.py --out mash/human_pool.jsonl \
      --in mash/kci_humanization_dataset_v0/kci_human_ko_cap100.jsonl \
      --in mash/kci_human_ko_v1.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--min-hangul", type=float, default=0.30)
    a = ap.parse_args()

    pool, seen, stats = [], set(), Counter()
    for path in a.inp:
        n0 = len(pool)
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            stats["총"] += 1
            aid = r.get("kci_article_id", "")
            text = (r.get("original_abstract") or "").strip()
            if not aid or aid in seen:
                stats["중복"] += 1
                continue
            if not (a.min_chars <= len(text) <= a.max_chars):
                stats["길이 밖"] += 1
                continue
            body = re.sub(r"\s", "", text)
            hr = len(HANGUL.findall(text)) / len(body) if body else 0
            if hr < a.min_hangul:
                stats["한글비 미달"] += 1
                continue
            field = r.get("research_field") or ("FIELD:" + (r.get("research_field_code") or "?"))
            seen.add(aid)
            pool.append({
                "kci_article_id": aid,
                "original_title": r.get("original_title", ""),
                "original_abstract": text,
                "research_field": field,
                "journal": r.get("journal") or r.get("journal_id", ""),
                "publication_date": r.get("publication_date") or r.get("registered", ""),
                "abstract_hangul_ratio": round(hr, 4),
                "origin": Path(path).name,
            })
        print(f"{Path(path).name}: +{len(pool)-n0:,}편")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n제외: " + " · ".join(f"{k} {v:,}" for k, v in stats.most_common() if k != "총"))
    print(f"인간 풀 {len(pool):,}편 → {out}")
    fields = Counter(r["research_field"] for r in pool)
    lens = sorted(len(r["original_abstract"]) for r in pool)
    print(f"  분야/코드 {len(fields)}종 (최다 {fields.most_common(1)[0][1]:,}편)"
          f" · 길이 중앙값 {lens[len(lens)//2]}자")


if __name__ == "__main__":
    main()
