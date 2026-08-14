#!/usr/bin/env python3
"""Stage 1 — D_pair 구성: OOF D₀ 필터 `D(x_human) < τ` ∧ `D(x_ai) > τ`.

- train: 반드시 OOF 점수 사용 (자기 채점 오염 방지 — notes/31 § Stage 0)
- dev  : 최종 D₀ 점수 사용 (dev 는 D₀ 학습에 안 들어갔으므로 깨끗)
- test : 필터하지 않는다 — 평가는 미필터 test 920 전체로 한다

산출물: loop/runs/stage1/dpair.jsonl.gz  (+ report.json)
  {doc_id, split, human_text, ai_text, d_human, d_ai, score_src}

사용:
  /workspace/.venv/bin/python3 loop/scripts/s1_build_dpair.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import data as D
from loop_lib import io_utils as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-min", type=int, default=0,
                    help="degenerate-case 바닥값 (스모크 전용). 엄격 필터 통과분이 이보다 적으면 "
                         "여유값(d_ai−d_human) 큰 순으로 채운다. 실운영(진짜 탐지기)에선 안 걸린다.")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    s0 = C.stage0_dir(cfg)
    tau = io.read_json(s0 / "detector_d0" / "tau.json")["tau"]
    oof = {r["doc_id"]: r for r in io.iter_jsonl(s0 / "oof_scores.jsonl.gz")}
    dev_scores = {r["doc_id"]: r for r in io.iter_jsonl(s0 / "scores_dev.jsonl.gz")}

    pairs = D.load_pairs(cfg, limit=a.limit)
    kept, stats = [], {"train": [0, 0], "dev": [0, 0]}  # split → [통과, 전체]
    rejected = {"train": [], "dev": []}  # 바닥값 채우기용 후보
    for r in pairs:
        if r["split"] == "test":
            continue
        src = oof if r["split"] == "train" else dev_scores
        s = src.get(r["doc_id"])
        if s is None:
            raise RuntimeError(f"{r['split']} 문서 {r['doc_id']} 의 점수가 없다 — s0 산출물과 데이터 불일치")
        stats[r["split"]][1] += 1
        rec = {
            "doc_id": r["doc_id"], "split": r["split"],
            "human_text": r["human_text"], "ai_text": r["ai_text"],
            "d_human": s["human"], "d_ai": s["ai"],
            "score_src": "oof" if r["split"] == "train" else "d0_final",
        }
        if s["human"] < tau and s["ai"] > tau:
            stats[r["split"]][0] += 1
            kept.append(rec)
        else:
            rejected[r["split"]].append(rec)

    if a.keep_min:
        for sp in ("train", "dev"):
            need = a.keep_min - stats[sp][0]
            if need > 0:
                extra = sorted(rejected[sp], key=lambda x: x["d_ai"] - x["d_human"], reverse=True)[:need]
                for e in extra:
                    e["score_src"] += "_floor"
                kept += extra
                print(f"[keep-min] {sp}: 엄격 통과 {stats[sp][0]} < {a.keep_min} → 여유값 {len(extra)}건 보충 (스모크)")

    out = C.stage1_dpair(cfg)
    io.write_jsonl(out, kept)
    report = {
        "tau": tau,
        "pass": {k: {"kept": v[0], "total": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None}
                 for k, v in stats.items()},
        "out": str(out),
    }
    io.write_json(out.parent / "report.json", report)
    print(f"D_pair: train {stats['train'][0]}/{stats['train'][1]} · dev {stats['dev'][0]}/{stats['dev'][1]} → {out}")


if __name__ == "__main__":
    main()
