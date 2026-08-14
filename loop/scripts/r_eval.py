#!/usr/bin/env python3
"""Round 단계 ④ — test 920 생성 + 전 지표 측정. **탐지기 재학습(⑤) 전에 실행한다.**

--round 0 은 SFT 직후 기준선 (MASH-ko 단일 라운드 재현의 전 단계).
ASR 규약: score ≤ τ ⇒ 인간 판정 ⇒ 공격 성공. (판정 규약: score > τ ⇒ AI)

산출물 (round{t}/eval/):
  gen_test.jsonl.gz   G_t 의 test 재서술 (greedy — 결정적)
  metrics.json        지표 전부 (판정표가 읽는 파일)
  details.jsonl.gz    문서별 점수·보존 내역
  report.md           사람용 요약
  ck_export/          카피킬러 업로드용 docx 배치 (AI 산출물만 — 인간 909는 기채점)

사용:
  python loop/scripts/r_eval.py --arm-config loop/configs/arms/main.yaml --round 1
  python loop/scripts/r_eval.py --round 0 --light        # SFT 기준선, 무거운 지표 생략
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from loop_lib import config as C
from loop_lib import data as D
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import metrics as M
from loop_lib import paraphraser as P


def asr(scores, tau):
    return float(np.mean(np.asarray(scores) <= tau))


def score_with(det_dir, texts_map, cfg, device):
    """texts_map: {이름: [텍스트]} → {이름: np.array}. 탐지기 하나 로드해 전부 채점."""
    model, tok = DET.load_detector(det_dir, device)
    out = {k: DET.predict_scores(model, tok, v, device, max_len=cfg.detector.max_len)
           for k, v in texts_map.items()}
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--light", action="store_true", help="simcse·ppl 생략 (스모크·빠른 확인)")
    ap.add_argument("--no-ck-export", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    t = a.round
    io.set_seed(cfg.seed)
    device = io.pick_device(a.device)
    edir = C.round_dir(cfg, t) / "eval"
    edir.mkdir(parents=True, exist_ok=True)

    pairs = D.load_pairs(cfg, limit=a.limit)
    test = D.by_split(pairs)["test"]
    h_texts = [r["human_text"] for r in test]
    x_texts = [r["ai_text"] for r in test]

    # ── ① 생성 (greedy) ─────────────────────────────────────────
    gen_path = edir / "gen_test.jsonl.gz"
    gen_dir = C.stage2_dir(cfg) / "best" if t == 0 else C.round_dir(cfg, t) / "dpo" / "final"
    if gen_path.exists() and not a.force:
        gen_rows = io.read_jsonl(gen_path)
        print(f"생성 있음 — 재사용 {len(gen_rows)}건: {gen_path}")
    else:
        model, tok = P.load(gen_dir, device)
        outs = P.paraphrase_texts(model, tok, x_texts, device, cfg.paraphraser,
                                  style=P.STYLE_HUMAN, greedy=cfg.eval.greedy, seed=cfg.seed,
                                  batch_size=cfg.eval.eval_batch_size,
                                  min_new_tokens=cfg.paraphraser.get("min_new_tokens", 0))
        gen_rows = [{"doc_id": r["doc_id"], "text": o[0]} for r, o in zip(test, outs)]
        io.write_jsonl(gen_path, gen_rows)
        del model
        n_empty = sum(1 for r in gen_rows if not r["text"].strip())
        if n_empty:
            print(f"⚠ 빈 생성물 {n_empty}/{len(gen_rows)}건 — 생성기 점검 필요")
        print(f"test {len(gen_rows)}건 생성 → {gen_path}")
    g_texts = [r["text"] for r in gen_rows]
    assert len(g_texts) == len(test)

    # ── ② 탐지기 채점 ───────────────────────────────────────────
    dt_dir = C.detector_in(cfg, max(t, 1))          # round 0·1 → D₀
    d0_dir = C.stage0_dir(cfg) / "detector_d0"
    tau_t = io.read_json(C.tau_of(dt_dir))["tau"]
    tau_0 = io.read_json(d0_dir / "tau.json")["tau"]

    s_dt = score_with(dt_dir, {"gen": g_texts, "human": h_texts, "xai": x_texts}, cfg, device)
    same_d = Path(dt_dir).resolve() == (d0_dir).resolve()
    s_d0 = s_dt if same_d else score_with(d0_dir, {"gen": g_texts, "human": h_texts, "xai": x_texts}, cfg, device)

    detectors = {
        "in_loop_dt": {"tau": tau_t, "gen_asr": asr(s_dt["gen"], tau_t),
                       "xai_asr": asr(s_dt["xai"], tau_t),
                       "human_fpr": float(np.mean(s_dt["human"] > tau_t)),
                       "gen_score_mean": float(np.mean(s_dt["gen"]))},
        "frozen_d0": {"tau": tau_0, "gen_asr": asr(s_d0["gen"], tau_0),
                      "xai_asr": asr(s_d0["xai"], tau_0),
                      "human_fpr": float(np.mean(s_d0["human"] > tau_0)),
                      "gen_score_mean": float(np.mean(s_d0["gen"]))},
    }
    an_dir = C.stage0_dir(cfg) / "anchor_koelectra"
    s_an = None
    if (an_dir / "tau.json").exists():
        tau_a = io.read_json(an_dir / "tau.json")["tau"]
        s_an = score_with(an_dir, {"gen": g_texts, "xai": x_texts}, cfg, device)
        detectors["frozen_koelectra"] = {"tau": tau_a, "gen_asr": asr(s_an["gen"], tau_a),
                                         "xai_asr": asr(s_an["xai"], tau_a),
                                         "gen_score_mean": float(np.mean(s_an["gen"]))}

    # ── ③ 문체·보존·분리도 ──────────────────────────────────────
    ref = M.style_stats(h_texts)
    dist_gen, per_gen = M.style_distance(g_texts, ref)
    dist_xai, _ = M.style_distance(x_texts, ref)

    pres = [M.preservation_row(r["human_text"], g, r["ai_text"])
            for r, g in zip(test, g_texts)]
    agg = {
        "len_ratio_mean": round(float(np.mean([p["len_ratio"] for p in pres])), 1),
        "nums_missing_rate": round(float(np.mean([p["nums_missing"] > 0 for p in pres])), 4),
        "nums_hallucinated_rate": round(float(np.mean([p["nums_hallucinated"] > 0 for p in pres])), 4),
        "dir_lost_rate": round(float(np.mean([bool(p["dir_lost"]) for p in pres])), 4),
        "overlap_vs_human_mean": round(float(np.mean([p["overlap_vs_human"] for p in pres])), 2),
        "overlap_vs_xai_mean": round(float(np.mean([p["overlap_vs_xai"] for p in pres])), 2),
        "lcs_vs_human_mean": round(float(np.mean([p["lcs_vs_human"] for p in pres])), 1),
    }
    if any("propn_recall" in p for p in pres):
        agg["propn_recall_mean"] = round(float(np.mean([p["propn_recall"] for p in pres if "propn_recall" in p])), 1)

    metrics = {
        "arm": cfg.arm.name, "round": t, "generator": str(gen_dir),
        "n_test": len(test),
        "detectors": detectors,
        "style": {"distance_gen": dist_gen, "distance_xai_baseline": dist_xai,
                  "per_feature_gen": per_gen},
        "preservation": agg,
        "separability": {
            "linear_cv_auc_gen_vs_human": M.linear_probe_cv(h_texts, g_texts),
            "length_auc_gen_vs_human": M.length_auc(h_texts, g_texts),
        },
        "feature_movement": M.feature_movement(h_texts, g_texts),
    }

    # ── ④ 무거운 지표 ───────────────────────────────────────────
    if not a.light and cfg.eval.enable_simcse:
        sims_h = M.simcse_similarity(g_texts, h_texts, cfg.eval.simcse_model, device)
        sims_x = M.simcse_similarity(g_texts, x_texts, cfg.eval.simcse_model, device)
        metrics["semantic"] = {"simcse_vs_human_mean": float(np.mean(sims_h)),
                               "simcse_vs_human_p10": float(np.percentile(sims_h, 10)),
                               "simcse_vs_xai_mean": float(np.mean(sims_x))}
    if not a.light and cfg.eval.enable_ppl:
        ppl_g = M.lm_perplexity(g_texts, cfg.eval.ppl_model, device)
        ppl_h = M.lm_perplexity(h_texts, cfg.eval.ppl_model, device)
        metrics["fluency"] = {"ppl_gen_mean": float(np.mean(ppl_g)),
                              "ppl_gen_median": float(np.median(ppl_g)),
                              "ppl_human_mean": float(np.mean(ppl_h))}

    # ── ⑤ 문서별 내역 + 카피킬러 export ─────────────────────────
    details = []
    for i, r in enumerate(test):
        row = {"doc_id": r["doc_id"], "d_t": float(s_dt["gen"][i]), "d_0": float(s_d0["gen"][i]),
               **pres[i]}
        if s_an is not None:
            row["d_anchor"] = float(s_an["gen"][i])
        details.append(row)
    io.write_jsonl(edir / "details.jsonl.gz", details)

    if not a.no_ck_export and cfg.eval.ck_export:
        try:
            export_ck(cfg, t, test, g_texts, edir / "ck_export")
        except ImportError as e:
            print(f"[warn] 카피킬러 export 생략 — python-docx 없음 ({e})")

    io.write_json(edir / "metrics.json", metrics)
    md = render_report(metrics)
    io.write_text(edir / "report.md", md)
    print(md)


def export_ck(cfg, t, test, g_texts, out):
    """카피킬러 업로드용 docx — stage0_export_for_detector.py 의 관례 준수:
    익명 일련번호 파일명(영숫자·언더바), manifest 는 배치 폴더 바깥, 문서 속성 비움."""
    from docx import Document

    out.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9]", "", f"{cfg.arm.name}R{t}") or f"R{t}"
    bs = cfg.eval.ck_batch_size
    rows = []
    for i, (r, text) in enumerate(zip(test, g_texts)):
        b = i // bs + 1
        bdir = out / f"batch{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{i+1:04d}.docx"
        doc = Document()
        for para in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            doc.add_paragraph(para)
        cp = doc.core_properties
        cp.author = cp.title = cp.comments = cp.last_modified_by = ""
        doc.save(str(bdir / name))
        rows.append({"file": name, "batch": f"batch{b:02d}", "pair_id": r["doc_id"],
                     "variant": f"loop_{cfg.arm.name}_r{t}", "model": f"kobart_r{t}",
                     "label": "ai", "chars": len(text)})
    import csv

    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "batch", "pair_id", "variant", "model", "label", "chars"])
        w.writeheader()
        w.writerows(rows)
    nb = rows[-1]["batch"] if rows else "batch00"
    print(f"카피킬러 export {len(rows)}건 → {out} ({nb} 까지) · manifest 는 업로드 금지")


def render_report(m):
    d = m["detectors"]
    L = [f"# round {m['round']} 평가 — arm={m['arm']}", "",
         f"- 생성기: `{m['generator']}` · test {m['n_test']}건", "",
         "| 탐지기 | τ | **생성물 ASR** | 원본 x_ai ASR | 비고 |", "| --- | --- | --- | --- | --- |"]
    note = {"in_loop_dt": "순환 — 증거 아님", "frozen_d0": "동결 앵커 1호", "frozen_koelectra": "동결 앵커 2호"}
    for k, v in d.items():
        L.append(f"| {k} | {v['tau']:.4f} | **{v['gen_asr']:.4f}** | {v['xai_asr']:.4f} | {note.get(k, '')} |")
    s = m["style"]
    L += ["", f"- 문체 z-거리: 생성물 **{s['distance_gen']:.3f}** vs x_ai {s['distance_xai_baseline']:.3f} (작을수록 인간 근접)"
          if s["distance_gen"] is not None else "- 문체 거리: kiwi 부재로 생략"]
    p = m["preservation"]
    L += [f"- 보존: 길이 {p['len_ratio_mean']}% · 수치 유실률 {p['nums_missing_rate']:.1%} · "
          f"환각률 {p['nums_hallucinated_rate']:.1%} · 방향어 소실률 {p['dir_lost_rate']:.1%}"
          + (f" · 고유명사 재현 {p['propn_recall_mean']}%" if "propn_recall_mean" in p else ""),
          f"- 복사: 인간 원문 12자 겹침 {p['overlap_vs_human_mean']}% · 입력 x_ai 겹침 {p['overlap_vs_xai_mean']}%",
          f"- 선형 분리도(CV AUC) {m['separability']['linear_cv_auc_gen_vs_human']:.4f} · "
          f"길이 AUC {m['separability']['length_auc_gen_vs_human']:.4f}"]
    if "semantic" in m:
        L.append(f"- 의미(SimCSE): 인간 대비 {m['semantic']['simcse_vs_human_mean']:.4f} · x_ai 대비 {m['semantic']['simcse_vs_xai_mean']:.4f}")
    if "fluency" in m:
        L.append(f"- PPL: 생성물 {m['fluency']['ppl_gen_median']:.1f}(중앙값) vs 인간 {m['fluency']['ppl_human_mean']:.1f}(평균)")
    fm = m.get("feature_movement")
    if fm:
        ai3 = " · ".join(f"{n}({w:+.2f})" for n, w in fm["ai_side"][:3])
        L.append(f"- 특징 이동(AI쪽 상위): {ai3}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
