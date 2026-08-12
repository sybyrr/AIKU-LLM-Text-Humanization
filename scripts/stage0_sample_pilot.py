#!/usr/bin/env python3
"""Stage 0 파일럿용 인간 글 표본 추출.

MASH Stage 1 검증에 쓸 인간 원문을 도메인별로 뽑는다.
분야 편향을 줄이려고 research_field 층화 후 라운드로빈으로 채운다.

사용:
    python3 stage0_sample_pilot.py --domain abstract \
        --in mash/kci_humanization_dataset_v0/kci_human_ko_cap100.jsonl \
        --out pilot/data/seed_abstract.jsonl --n 50
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# 길이 하한: 탐지기 신뢰도가 떨어지는 짧은 글 배제 (notes/README 쟁점 3, 카피킬러 300자 이슈)
MIN_CHARS = 300
# 길이 상한: CPU 생성 시간과 llama-server ctx 한계를 고려
MAX_CHARS = 1200


def load(path, text_field, id_field, field_field):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get(text_field) or "").strip()
            if not MIN_CHARS <= len(text) <= MAX_CHARS:
                continue
            rows.append(
                {
                    "doc_id": str(r[id_field]),
                    "text": text,
                    "stratum": (r.get(field_field) or "UNKNOWN").strip(),
                    "meta": {k: v for k, v in r.items() if k != text_field},
                }
            )
    return rows


def stratified(rows, n, seed):
    rnd = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["stratum"]].append(r)
    for b in buckets.values():
        rnd.shuffle(b)
    # 큰 층부터 라운드로빈 — 층 수가 n보다 많아도 고르게 퍼진다
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    picked, i = [], 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True, help="abstract | argument")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-field", default="original_abstract")
    ap.add_argument("--id-field", default="kci_article_id")
    ap.add_argument("--field-field", default="research_field")
    a = ap.parse_args()

    rows = load(a.inp, a.text_field, a.id_field, a.field_field)
    picked = stratified(rows, a.n, a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(
                json.dumps(
                    {
                        "pair_id": f"{a.domain}-{r['doc_id']}",
                        "domain": a.domain,
                        "source_document_id": r["doc_id"],
                        "stratum": r["stratum"],
                        "human_text": r["text"],
                        "human_chars": len(r["text"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    lens = sorted(len(r["text"]) for r in picked)
    strata = len({r["stratum"] for r in picked})
    print(f"후보 {len(rows)}건 → 표본 {len(picked)}건 ({strata}개 층)")
    print(f"길이 min/median/max: {lens[0]} / {lens[len(lens)//2]} / {lens[-1]}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
