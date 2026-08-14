#!/usr/bin/env python3
"""Round 단계 ② — preference pair 구성 + ref logp 사전계산.

규칙 (notes/31 § Round 2):
  y_w (chosen):
    arm.anchor = human → **원래 인간 초록** (인간 앵커, 라운드 내내 고정)
    arm.anchor = self  → 후보 중 D_t 점수가 가장 낮은 자기 생성물 (ablation)
  y_l (rejected): D_t(y_l) > τ_t 인 hard negative, 점수 높은 순 max_pairs_per_doc 개
  부족한 문서는 τ를 낮추지 않고 needs_more 로 보고 → r_generate --top-up 이 증량.
  (n_candidates_max 까지 늘려도 없으면 그 문서는 이 라운드에서 제외)

ref logp: G_{t-1}(=DPO 정책 초기값)로 (y_w, y_l) 의 Σlog p 를 지금 계산해 저장
— r_dpo 는 ref 모델 없이 돈다.

--fallback-top1: hard negative 가 없는 문서에서 최고점 후보를 그냥 쓴다.
  실운영에서는 쓰지 말 것 (ΔD≈0 쌍은 라벨 잡음 — MASH Appendix A.2). 스모크 전용.
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from loop_lib import config as C
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import paraphraser as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fallback-top1", action="store_true", help="스모크 전용 — 실운영 금지")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    t = a.round
    io.set_seed(cfg.seed)
    device = io.pick_device(a.device)
    rdir = C.round_dir(cfg, t)
    out_path = rdir / "prefs.jsonl.gz"
    if out_path.exists() and not a.force:
        print(f"prefs 있음 — 재사용: {out_path} (--force 로 재생성)")
        return

    rows = [r for r in io.read_jsonl(C.stage1_dpair(cfg)) if r["split"] == "train"]
    if a.limit:
        rows = rows[: a.limit]
    by_doc = {r["doc_id"]: r for r in rows}

    cands = collections.defaultdict(list)
    for r in io.iter_jsonl(rdir / "candidates.jsonl.gz"):
        if r["doc_id"] in by_doc:
            cands[r["doc_id"]].append(r["text"])

    # ── D_t 로 후보 + y_w 후보(인간 원문) 채점 ─────────────────
    det_dir = C.detector_in(cfg, t)
    tau = io.read_json(C.tau_of(det_dir))["tau"]
    dmodel, dtok = DET.load_detector(det_dir, device)
    print(f"D_{t} = {det_dir}  τ_{t} = {tau:.4f}  문서 {len(cands)}개")

    flat, keys = [], []
    for d, texts in cands.items():
        for i, text in enumerate(texts):
            flat.append(text)
            keys.append((d, i))
    scores = DET.predict_scores(dmodel, dtok, flat, device, max_len=cfg.detector.max_len)
    sc = collections.defaultdict(list)
    for (d, i), s in zip(keys, scores):
        sc[d].append((float(s), cands[d][i]))
    del dmodel
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── pair 구성 ───────────────────────────────────────────────
    anchor = cfg.arm.anchor
    prefs, needs_more, no_more = [], [], []
    for d, scored in sc.items():
        hard = sorted([x for x in scored if x[0] > tau], key=lambda x: -x[0])
        if not hard and a.fallback_top1:
            hard = sorted(scored, key=lambda x: -x[0])[:1]
        if anchor == "human":
            y_w, d_w = by_doc[d]["human_text"], None
        elif anchor == "self":
            d_w, y_w = min(scored, key=lambda x: x[0])
        else:
            raise ValueError(f"arm.anchor={anchor}")
        picked = [(s, y) for s, y in hard if y != y_w][: cfg.rounds.max_pairs_per_doc]
        if not picked:
            (needs_more if len(scored) < cfg.rounds.n_candidates_max else no_more).append(d)
            continue
        for d_l, y_l in picked:
            prefs.append({"doc_id": d, "x_ai": by_doc[d]["ai_text"], "y_w": y_w, "y_l": y_l,
                          "d_w": d_w, "d_l": d_l})

    # ── ref logp (G_{t-1}) ──────────────────────────────────────
    if prefs:
        gmodel, gtok = P.load(C.generator_in(cfg, t), device)
        gmodel.eval()
        bs = cfg.dpo.batch_size
        from loop_lib.dpo import make_collate

        collate = make_collate(gtok, cfg.paraphraser)
        with torch.no_grad():
            for s in range(0, len(prefs), bs):
                chunk = prefs[s : s + bs]
                enc, lab_w, lab_l, _, _ = collate([{**c, "ref_logp_w": 0.0, "ref_logp_l": 0.0}
                                                   for c in chunk])
                style = torch.full((len(chunk),), P.STYLE_HUMAN, dtype=torch.long, device=device)
                lw, ll = gmodel.pair_logprobs(enc["input_ids"].to(device),
                                              enc["attention_mask"].to(device), style,
                                              lab_w.to(device), lab_l.to(device))
                for c, w, l in zip(chunk, lw.tolist(), ll.tolist()):
                    c["ref_logp_w"], c["ref_logp_l"] = w, l
        del gmodel

    io.write_jsonl(out_path, prefs)
    report = {
        "round": t, "arm": cfg.arm.name, "tau": tau,
        "docs_total": len(by_doc), "docs_with_cands": len(sc),
        "docs_covered": len({p["doc_id"] for p in prefs}), "n_prefs": len(prefs),
        "needs_more": needs_more, "given_up": len(no_more),
        "fallback_top1": a.fallback_top1,
    }
    io.write_json(rdir / "prefs_report.json", report)
    print(f"prefs {len(prefs)}건 · 커버 {report['docs_covered']}/{len(by_doc)}문서 "
          f"· 증량 필요 {len(needs_more)} · 포기 {len(no_more)} → {out_path}")
    if needs_more:
        print(f"→ r_generate.py --round {t} --top-up 후 r_build_prefs.py --round {t} --force 재실행")


if __name__ == "__main__":
    main()
