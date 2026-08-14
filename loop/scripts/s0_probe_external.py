#!/usr/bin/env python3
"""외부 생성물(generate.py 산출)을 동결 탐지기로 채점 — OOD·P4·모드B 공용.

세 용도, 한 도구 (notes/31 § 생성기 다양화·재서술 편향·P4):
  OOD 프로브  --ref-source test       --role detect  held-out 생성기가 만든 x_ai 를
              D₀ 가 잡는가 (TPR). "D₀ 가 AI 를 잡나 Qwen3 지문을 잡나"
  P4 베이스라인 --ref-source test      --role evade   지문 명시 프롬프트가 x_ai 를
              인간화하는가 (ASR). 학습된 humanizer 와 같은 test x_ai 대상
  모드B 평가  --ref-source eval_bodies --role detect  '처음부터 쓴 AI 초록'을 D₀ 가 잡는가
              (재서술 편향 ⓑ 검증)

역할별 해석:
  detect → gen 을 AI 로 보고 TPR@τ (높을수록 탐지기가 튼튼)
  evade  → gen 을 인간화 시도로 보고 ASR = P(score ≤ τ) (높을수록 회피 성공)
둘은 같은 점수의 앞뒤다 — 둘 다 출력해 해석은 읽는 사람에게 맡긴다.

산출물: loop/runs/probes/<name>/report.{json,md} · scores.jsonl.gz · (선택) ck_export/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from loop_lib import config as C
from loop_lib import data as D
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import metrics as M


def load_refs(cfg, source):
    """doc_id → {"human":…, "xai":… (있으면)}. 생성물의 doc_id 와 조인용."""
    if source == "none":
        return {}
    if source == "test":
        pairs = D.by_split(D.load_pairs(cfg))["test"]
        return {r["doc_id"]: {"human": r["human_text"], "xai": r["ai_text"]} for r in pairs}
    if source == "eval_bodies":
        out = {}
        for r in io.iter_jsonl(C.rp(cfg.data.eval_bodies)):
            out[f"eval-{r['kci_article_id']}"] = {"human": r["original_abstract"]}
        return out
    raise ValueError(source)


def norm_id(doc_id):
    """generate.py 산출의 doc_id 접두(eval-/abstract-)를 참조 키와 맞춘다."""
    return doc_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--gen", required=True, help="generate.py 산출 jsonl")
    ap.add_argument("--name", required=True, help="probes/<name>/ 산출 폴더")
    ap.add_argument("--ref-source", choices=["test", "eval_bodies", "none"], default="test")
    ap.add_argument("--role", choices=["detect", "evade"], default="detect")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ck-export", action="store_true")
    ap.add_argument("--light", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    device = io.pick_device(a.device)
    outdir = C.runs_dir(cfg) / "probes" / a.name
    outdir.mkdir(parents=True, exist_ok=True)

    gens = [g for g in io.iter_jsonl(C.rp(a.gen)) if g.get("text")]
    refs = load_refs(cfg, a.ref_source)
    g_texts = [g["text"] for g in gens]
    ids = [norm_id(g["doc_id"]) for g in gens]
    matched = [i for i, d in enumerate(ids) if d in refs] if refs else []
    print(f"생성물 {len(gens)}건 · 참조 매칭 {len(matched)}/{len(gens)} (source={a.ref_source})")

    # ── 동결 탐지기 채점 (D₀ + 앵커, 루프에 절대 안 들어간 것) ──
    d0 = C.stage0_dir(cfg) / "detector_d0"
    tau0 = io.read_json(d0 / "tau.json")["tau"]
    dm, dtok = DET.load_detector(d0, device)
    g_scores = DET.predict_scores(dm, dtok, g_texts, device, max_len=cfg.detector.max_len)
    del dm

    result = {"name": a.name, "gen": str(a.gen), "role": a.role, "ref_source": a.ref_source,
              "n_gen": len(gens), "n_matched": len(matched)}
    tpr = float(np.mean(g_scores > tau0))
    result["frozen_d0"] = {"tau": tau0, "gen_score_mean": float(np.mean(g_scores)),
                           "tpr_as_ai": tpr, "asr_as_evasion": 1.0 - tpr}

    an = C.stage0_dir(cfg) / "anchor_koelectra"
    if (an / "tau.json").exists():
        taua = io.read_json(an / "tau.json")["tau"]
        am, atok = DET.load_detector(an, device)
        a_scores = DET.predict_scores(am, atok, g_texts, device, max_len=cfg.detector.max_len)
        del am
        tpra = float(np.mean(a_scores > taua))
        result["frozen_koelectra"] = {"tau": taua, "tpr_as_ai": tpra, "asr_as_evasion": 1.0 - tpra}
    else:
        a_scores = None

    # ── 참조가 있으면 문체·보존·분리도 ─────────────────────────
    if matched:
        m_g = [g_texts[i] for i in matched]
        m_h = [refs[ids[i]]["human"] for i in matched]
        ref_stats = M.style_stats(m_h)
        dist, _ = M.style_distance(m_g, ref_stats)
        result["style_distance_gen"] = dist
        result["separability"] = {
            "linear_cv_auc_vs_human": M.linear_probe_cv(m_h, m_g),
            "length_auc_vs_human": M.length_auc(m_h, m_g),
        }
        has_xai = all("xai" in refs[ids[i]] for i in matched)
        pres = [M.preservation_row(refs[ids[i]]["human"], g_texts[i],
                                   refs[ids[i]].get("xai") if has_xai else None)
                for i in matched]
        result["preservation"] = {
            "len_ratio_mean": round(float(np.mean([p["len_ratio"] for p in pres])), 1),
            "nums_missing_rate": round(float(np.mean([p["nums_missing"] > 0 for p in pres])), 4),
            "dir_lost_rate": round(float(np.mean([bool(p["dir_lost"]) for p in pres])), 4),
            "overlap_vs_human_mean": round(float(np.mean([p["overlap_vs_human"] for p in pres])), 2),
        }
        result["feature_movement"] = M.feature_movement(m_h, m_g)

    # ── 문서별 점수 ─────────────────────────────────────────────
    rows = []
    for i, g in enumerate(gens):
        row = {"doc_id": g["doc_id"], "d0": float(g_scores[i]), "chars": len(g["text"])}
        if a_scores is not None:
            row["d_anchor"] = float(a_scores[i])
        rows.append(row)
    io.write_jsonl(outdir / "scores.jsonl.gz", rows)

    if a.ck_export:
        try:
            _ck(cfg, gens, outdir / "ck_export", a.name)
        except ImportError as e:
            print(f"[warn] ck export 생략 — {e}")

    io.write_json(outdir / "report.json", result)
    io.write_text(outdir / "report.md", _md(result))
    print(_md(result))


def _ck(cfg, gens, out, name):
    import csv
    import re

    from docx import Document

    out.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9]", "", name) or "probe"
    bs = cfg.eval.ck_batch_size
    rows = []
    for i, g in enumerate(gens):
        b = i // bs + 1
        bdir = out / f"batch{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        fn = f"{prefix}_{i+1:04d}.docx"
        doc = Document()
        for para in g["text"].replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            doc.add_paragraph(para)
        cp = doc.core_properties
        cp.author = cp.title = cp.comments = cp.last_modified_by = ""
        doc.save(str(bdir / fn))
        rows.append({"file": fn, "batch": f"batch{b:02d}", "pair_id": g["doc_id"],
                     "variant": name, "model": g.get("model", "?"), "label": "ai",
                     "chars": len(g["text"])})
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "batch", "pair_id", "variant", "model", "label", "chars"])
        w.writeheader()
        w.writerows(rows)
    print(f"ck export {len(rows)}건 → {out}")


def _md(r):
    d = r["frozen_d0"]
    L = [f"# 외부 프로브 — {r['name']} (role={r['role']})", "",
         f"- 생성물 {r['n_gen']}건 · 참조 매칭 {r['n_matched']} · source={r['ref_source']}",
         f"- **동결 D₀** τ={d['tau']:.4f} · 점수평균 {d['gen_score_mean']:.4f}",
         f"    - AI 로 볼 때 TPR **{d['tpr_as_ai']:.4f}** · 인간화로 볼 때 ASR **{d['asr_as_evasion']:.4f}**"]
    if "frozen_koelectra" in r:
        k = r["frozen_koelectra"]
        L.append(f"- 동결 KoELECTRA τ={k['tau']:.4f} · TPR {k['tpr_as_ai']:.4f} · ASR {k['asr_as_evasion']:.4f}")
    if "style_distance_gen" in r and r["style_distance_gen"] is not None:
        L.append(f"- 문체 z-거리 {r['style_distance_gen']:.3f}")
    if "preservation" in r:
        p = r["preservation"]
        L.append(f"- 보존: 길이 {p['len_ratio_mean']}% · 수치유실 {p['nums_missing_rate']:.1%} · "
                 f"방향어소실 {p['dir_lost_rate']:.1%} · 인간 12자겹침 {p['overlap_vs_human_mean']}%")
    if r.get("separability"):
        s = r["separability"]
        L.append(f"- 선형 CV AUC {s['linear_cv_auc_vs_human']:.4f} · 길이 AUC {s['length_auc_vs_human']:.4f}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
