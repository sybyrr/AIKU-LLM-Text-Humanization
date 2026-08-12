#!/usr/bin/env python3
"""최종 pair 데이터셋 조립 — 카피킬러 판정을 통과한 (인간, AI) 짝만 모은다.

MASH Stage 1 의 게이트 두 개를 그대로 적용한다 (`MASH.md:102`):
  ① D(x_human) < τ   인간 원문이 인간으로 판정
  ② D(x_ai)   ≥ τ    AI 재서술이 AI 로 판정
여기에 생성 실패만 걸러내는 "깨짐" 검사를 더한다(내용 보존 필터는 논문에 없어 쓰지 않는다).

분할 주의: **원문 단위로 나눈다.** 1:1 구성이라 지금은 pair = 원문이지만,
나중에 arm 을 늘리면 같은 원문에서 나온 쌍들이 서로 다른 split 에 흩어져
누수가 생긴다. 지금부터 원문 기준으로 나눠 두면 그 위험이 없다.

필드는 인수받은 v0 README 의 "Pair 주요 field" 규약을 따른다.

사용:
  python3 build_final_dataset.py --out mash/pairs_v1.jsonl
"""
import argparse
import collections
import json
import random
from pathlib import Path


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mash/pairs_v1.jsonl")
    ap.add_argument("--scores", action="append",
                    default=["pilot/scores/copykiller_full.jsonl",
                             "pilot/scores/copykiller_all.jsonl"])
    ap.add_argument("--seed-file", default="mash/seed_full.jsonl")
    ap.add_argument("--gen", default="mash/gen_full_p3b_clean.jsonl")
    ap.add_argument("--pool", default="mash/human_pool.jsonl")
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--split", default="8:1:1", help="train:dev:test (원문 단위)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    hum_score, ai_score = {}, {}
    for p in a.scores:
        if not Path(p).exists():
            continue
        for r in load_jsonl(p):
            # 파일럿 점수 파일에는 다른 프롬프트(P1·P2·P3a·P3c)의 AI 도 들어 있다 — 제외
            if r["label"] == "human" and r.get("variant") == "human":
                hum_score[r["pair_id"]] = r["ck_ai_pct"]
            elif r["label"] == "ai" and r.get("variant") == "P3b":
                ai_score[r["pair_id"]] = r["ck_ai_pct"]

    human = {r["pair_id"]: r for r in load_jsonl(a.seed_file)}
    gen = {g["doc_id"]: g for g in load_jsonl(a.gen) if g.get("text")}
    meta = {r["kci_article_id"]: r for r in load_jsonl(a.pool)}

    rows, stats = [], collections.Counter()
    for pid, g in gen.items():
        stats["생성 성공"] += 1
        if pid not in hum_score or pid not in ai_score:
            stats["미검사"] += 1
            continue
        if hum_score[pid] >= a.tau:
            stats["인간측 오탐"] += 1
            continue
        if ai_score[pid] < a.tau:
            stats["flip 실패"] += 1
            continue
        h = human[pid]
        m = meta.get(h["source_document_id"], {})
        rows.append({
            "id": pid,
            "human_text": h["human_text"],
            "ai_text": g["text"],
            "generator": "Qwen3-8B-Q4_K_M",
            "prompt_version": "P3b",
            "research_field": m.get("research_field", h.get("stratum", "")),
            "journal": m.get("journal", ""),
            "publication_date": m.get("publication_date", ""),
            "original_title": m.get("original_title", ""),
            "publisher": m.get("publisher", ""),
            "kci_article_id": h["source_document_id"],
            "detector": "copykiller",
            "ck_human_pct": hum_score[pid],
            "ck_ai_pct": ai_score[pid],
        })
        stats["유효쌍"] += 1

    # 원문 단위 분할 (1:1 이라 지금은 쌍 단위와 같지만, arm 확장 대비)
    docs = sorted({r["kci_article_id"] for r in rows})
    random.Random(a.seed).shuffle(docs)
    w = [int(x) for x in a.split.split(":")]
    n1 = len(docs) * w[0] // sum(w)
    n2 = n1 + len(docs) * w[1] // sum(w)
    split = {d: ("train" if i < n1 else "dev" if i < n2 else "test")
             for i, d in enumerate(docs)}
    for r in rows:
        r["split"] = split[r["kci_article_id"]]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(" · ".join(f"{k} {v:,}" for k, v in stats.most_common()))
    print(f"\n{len(rows):,}쌍 → {out}")
    sc = collections.Counter(r["split"] for r in rows)
    print("  분할: " + " · ".join(f"{k} {sc[k]:,}" for k in ("train", "dev", "test")))
    hl = sorted(len(r["human_text"]) for r in rows)
    al = sorted(len(r["ai_text"]) for r in rows)
    print(f"  길이 중앙값: 인간 {hl[len(hl)//2]}자 · AI {al[len(al)//2]}자")
    fields = collections.Counter(r["research_field"] for r in rows)
    print(f"  분야 {len(fields)}종 · 최다 {fields.most_common(1)[0][0]} {fields.most_common(1)[0][1]:,}쌍")


if __name__ == "__main__":
    main()
