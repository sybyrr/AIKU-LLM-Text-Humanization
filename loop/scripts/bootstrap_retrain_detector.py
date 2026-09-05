#!/usr/bin/env python3
"""가져온 팀 DPO 생성기 G1의 출력으로 첫 적응 탐지기 D1을 학습한다.

이 단계가 기존 팀 DPO(G1)와 사용자 루프의 round1 DPO(G2)를 연결한다.
학습 buffer는 human 50%, AI 50%(원본 AI와 G1 출력을 균등)이며 dev-cal과
dev-gate를 분리해 임계값 캘리브레이션과 붕괴 판정을 독립적으로 수행한다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from loop_lib import config as C
from loop_lib import data as D
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import paraphraser as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=str(C.LOOP_DIR / "configs" / "arms" / "team_news.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    io.set_seed(cfg.seed)
    device = io.pick_device(a.device)
    pairs = D.load_pairs(cfg, limit=a.limit)
    by = D.by_split(pairs)
    cal, gate_docs = D.dev_halves(by["dev"])
    if not cal or not gate_docs:
        raise RuntimeError("dev-cal/dev-gate가 비었다 — --limit을 늘려라")

    replay_path = C.rp(cfg.arm.bootstrap_replay)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    if a.force and replay_path.exists():
        replay_path.unlink()
    generated = io.read_jsonl(replay_path) if replay_path.exists() else []
    if len({r.get("doc_id") for r in generated}) != len(generated):
        raise RuntimeError(f"기존 replay에 중복 doc_id가 있다: {replay_path}")
    train_by_id = {r["doc_id"]: r for r in by["train"]}
    unknown = {r.get("doc_id") for r in generated} - set(train_by_id)
    if unknown:
        raise RuntimeError(f"기존 replay에 현재 train 밖 doc_id가 있다: {sorted(unknown)[:3]}")
    have = {r["doc_id"] for r in generated}
    todo = [r for r in by["train"] if r["doc_id"] not in have]
    if not todo:
        print(f"G1 replay 있음 — 재사용 {len(generated)}건: {replay_path}")
    else:
        model, tok = P.load(C.generator_in(cfg, 1), device)
        chunk_size = max(128, cfg.eval.eval_batch_size * 8)
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start:start + chunk_size]
            outs = P.paraphrase_texts(
                model, tok, [r["ai_text"] for r in chunk], device, cfg.paraphraser,
                style=P.STYLE_HUMAN, greedy=True, seed=cfg.seed + start,
                batch_size=cfg.eval.eval_batch_size,
                min_new_tokens=cfg.paraphraser.get("min_new_tokens", 0),
            )
            new = [{"doc_id": r["doc_id"], "text": o[0]} for r, o in zip(chunk, outs)]
            if any(not r["text"] for r in new):
                raise RuntimeError("G1 replay에 빈 생성물이 있다")
            io.append_jsonl(replay_path, new)
            generated.extend(new)
            print(f"G1 replay {len(generated):,}/{len(by['train']):,}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"G1 train replay 완료 {len(generated):,}건 → {replay_path}")

    if len(generated) != len(by["train"]):
        raise RuntimeError(f"replay 수 불일치: {len(generated)} != train {len(by['train'])}")
    gen_by = {r["doc_id"]: r["text"] for r in generated}
    if set(gen_by) != {r["doc_id"] for r in by["train"]}:
        raise RuntimeError("replay doc_id가 train split과 일치하지 않는다")

    humans = [r["human_text"] for r in by["train"]]
    orig_ai = [r["ai_text"] for r in by["train"]]
    para = [gen_by[r["doc_id"]] for r in by["train"]]
    n = len(humans)
    texts = humans + orig_ai + para
    labels = [0] * n + [1] * (2 * n)
    weights = [0.5 / n] * n + [0.25 / n] * n + [0.25 / n] * n
    info = {"human": n, "orig_ai": n, "para_bootstrap": n, "mode": "bootstrap_full"}

    model_dir = C.detector_in(cfg, 1)
    ddir = model_dir.parent if model_dir.name == "model" else model_dir
    tau_path = ddir / "tau.json"
    if tau_path.exists() and not a.force:
        model, tok = DET.load_detector(model_dir, device)
        tau = io.read_json(tau_path)["tau"]
        print(f"D1 있음 — 재사용: {model_dir}")
    else:
        factory = DET.make_factory(cfg.detector.model, tiny=a.tiny, init_from=None)
        cal_texts, cal_labels, _ = D.texts_labels(cal)
        model, tok, hist = DET.train_detector(
            factory, texts, labels, cfg.detector, device,
            dev_texts=cal_texts, dev_labels=cal_labels, sample_weights=weights,
            epochs=a.epochs, log_prefix=f"D1-{cfg.arm.name}",
        )
        cal_h = DET.predict_scores(model, tok, [r["human_text"] for r in cal], device,
                                   max_len=cfg.detector.max_len)
        tau = DET.calibrate_tau(cal_h, cfg.detector.fpr_target)
        DET.save_detector(model, tok, model_dir, {
            "round": 1, "arm": cfg.arm.name, "buffer": info,
            "reinit": True, "history": hist, "bootstrap_from": str(C.generator_in(cfg, 1)),
        })
        io.write_json(tau_path, {
            "tau": tau, "fpr_target": cfg.detector.fpr_target,
            "calibrated_on": "dev_cal_human", "n_human": len(cal),
        })

    ml = cfg.detector.max_len
    g_h = DET.predict_scores(model, tok, [r["human_text"] for r in gate_docs], device, max_len=ml)
    g_a = DET.predict_scores(model, tok, [r["ai_text"] for r in gate_docs], device, max_len=ml)
    fpr = DET.fpr_at(g_h, tau)
    tpr = DET.tpr_at(g_a, tau)
    collapsed = bool(fpr > cfg.gate.fpr_max or tpr < cfg.gate.tpr_min)
    gate = {
        "round": 1, "arm": cfg.arm.name, "tau": tau,
        "gate_human_fpr": fpr, "gate_orig_ai_tpr": tpr,
        "fpr_max": cfg.gate.fpr_max, "tpr_min": cfg.gate.tpr_min,
        "collapsed": collapsed, "n_gate": len(gate_docs), "buffer": info,
    }
    io.write_json(ddir / "gate.json", gate)
    verdict = "붕괴 — G2 DPO 진행 금지" if collapsed else "정상 — G2 DPO 진행 가능"
    print(f"D1: τ={tau:.6f} · gate FPR={fpr:.4f} · orig-AI TPR={tpr:.4f} → {verdict}")
    if collapsed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
