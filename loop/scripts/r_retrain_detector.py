#!/usr/bin/env python3
"""Round 단계 ⑤ — D_{t+1} 재학습 + 붕괴 게이트. **r_eval 이후에 실행한다.**

절차 (notes/31 § Round 5–6):
  A. G_t 로 train 전 문서의 x_ai 를 greedy 재서술 → replay_gen.jsonl.gz
  B. replay buffer 조립 (arm.replay: full | latest_only) — class-balanced 가중치
  C. D_{t+1} 학습 — arm.detector_reinit ? 사전학습 체크포인트 재초기화 : D_t 에서 continual
  D. τ_{t+1} = dev-cal 인간 FPR 5% 지점
  E. 게이트 — dev-gate 에서 FPR > fpr_max 또는 원본 AI TPR < tpr_min ⇒ 붕괴 기록
     (붕괴면 그 라운드의 DPO 신호 무효 — run_arm.sh 가 gate.json 을 읽고 중단)

사용:
  python loop/scripts/r_retrain_detector.py --arm-config loop/configs/arms/main.yaml --round 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from loop_lib import config as C
from loop_lib import data as D
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import paraphraser as P
from loop_lib import replay as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="초소형 탐지기 (스모크)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    t = a.round
    io.set_seed(cfg.seed + t)
    device = io.pick_device(a.device)
    ddir = C.round_dir(cfg, t) / "detector"
    ddir.mkdir(parents=True, exist_ok=True)

    pairs = D.load_pairs(cfg, limit=a.limit)
    by = D.by_split(pairs)
    cal, gate_docs = D.dev_halves(by["dev"])

    # ── A. G_t 재서술 (replay 소스) ─────────────────────────────
    rg_path = R.replay_gen_path(cfg, t)
    if rg_path.exists() and not a.force:
        print(f"replay 생성물 있음 — 재사용: {rg_path}")
    else:
        gmodel, gtok = P.load(C.round_dir(cfg, t) / "dpo" / "final", device)
        texts = [r["ai_text"] for r in by["train"]]
        outs = P.paraphrase_texts(gmodel, gtok, texts, device, cfg.paraphraser,
                                  style=P.STYLE_HUMAN, greedy=cfg.eval.greedy, seed=cfg.seed,
                                  batch_size=cfg.eval.eval_batch_size,
                                  min_new_tokens=cfg.paraphraser.get("min_new_tokens", 0))
        io.write_jsonl(rg_path, [{"doc_id": r["doc_id"], "text": o[0]}
                                 for r, o in zip(by["train"], outs)])
        del gmodel
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"G_{t} train 재서술 {len(by['train'])}건 → {rg_path}")

    # ── B~D. buffer 조립 → 학습 → τ ─────────────────────────────
    if (ddir / "tau.json").exists() and not a.force:
        print(f"D_{t+1} 있음 — 재사용: {ddir}")
        model, tok = DET.load_detector(ddir / "model", device)
        tau = io.read_json(ddir / "tau.json")["tau"]
    else:
        texts, labels, weights, info = R.build_buffer(cfg, by["train"], t)
        print(f"replay buffer: {info}")
        init_from = None if cfg.arm.detector_reinit else C.detector_in(cfg, t)
        if init_from:
            print(f"continual — {init_from} 에서 이어서 학습")
        factory = DET.make_factory(cfg.detector.model, tiny=a.tiny, init_from=init_from)
        cal_texts, cal_labels, _ = D.texts_labels(cal)
        model, tok, hist = DET.train_detector(
            factory, texts, labels, cfg.detector, device,
            dev_texts=cal_texts, dev_labels=cal_labels, sample_weights=weights,
            epochs=a.epochs, log_prefix=f"D{t+1}-{cfg.arm.name}",
        )
        cal_h = DET.predict_scores(model, tok, [r["human_text"] for r in cal], device,
                                   max_len=cfg.detector.max_len)
        tau = DET.calibrate_tau(cal_h, cfg.detector.fpr_target)
        DET.save_detector(model, tok, ddir / "model",
                          {"round": t + 1, "arm": cfg.arm.name, "buffer": info,
                           "reinit": cfg.arm.detector_reinit, "history": hist})
        io.write_json(ddir / "tau.json", {"tau": tau, "fpr_target": cfg.detector.fpr_target,
                                          "calibrated_on": "dev_cal_human", "n_human": len(cal)})

    # ── E. 게이트 (dev-gate — 캘리브레이션과 분리된 인간셋) ─────
    ml = cfg.detector.max_len
    g_h = DET.predict_scores(model, tok, [r["human_text"] for r in gate_docs], device, max_len=ml)
    g_a = DET.predict_scores(model, tok, [r["ai_text"] for r in gate_docs], device, max_len=ml)
    fpr = DET.fpr_at(g_h, tau)
    tpr = DET.tpr_at(g_a, tau)
    collapsed = bool(fpr > cfg.gate.fpr_max or tpr < cfg.gate.tpr_min)
    gate = {"round": t + 1, "arm": cfg.arm.name, "tau": tau,
            "gate_human_fpr": fpr, "gate_orig_ai_tpr": tpr,
            "fpr_max": cfg.gate.fpr_max, "tpr_min": cfg.gate.tpr_min,
            "collapsed": collapsed, "n_gate": len(gate_docs)}
    io.write_json(ddir / "gate.json", gate)
    verdict = "⚠ 탐지기 붕괴 — 이 라운드 DPO 신호 무효 (MASH Appendix D 참조)" if collapsed else "정상"
    print(f"D_{t+1}: τ={tau:.4f} · gate FPR {fpr:.4f} (한도 {cfg.gate.fpr_max}) · "
          f"원본 AI TPR {tpr:.4f} (바닥 {cfg.gate.tpr_min}) → {verdict}")


if __name__ == "__main__":
    main()
