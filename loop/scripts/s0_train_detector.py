#!/usr/bin/env python3
"""Stage 0 — 대리 탐지기 D₀ 학습 + 동결 앵커 + τ₀ 캘리브레이션.

산출물 (loop/runs/stage0/):
  oof_scores.jsonl.gz     train split 의 k-fold OOF 점수 (Stage 1 필터 전용)
  detector_d0/            최종 D₀ (train 전체로 학습, dev-cal AUC 최고 에폭) + tau.json
                          ← 이 폴더가 곧 "동결 앵커 1호". 라운드 산출물은 arms/ 아래라 절대 덮이지 않는다.
  anchor_koelectra/       동결 앵커 2호 (다른 아키텍처) + 자체 tau.json
  scores_dev.jsonl.gz     D₀ 의 dev 점수 (cal/gate 표기 포함)
  scores_test.jsonl.gz    D₀ 의 test 점수 (라운드별 "고정 D₀ ASR" 의 기준선)
  report.json / report.md

검증 기준 (notes/31 실측):
  선형 베이스라인(TF-IDF char 2-4gram + LR) test AUC ≈ 0.993 — D₀ 가 이걸 못 넘으면 학습이 잘못된 것.
  길이 단독 AUC ≈ 0.5 — 길이 지름길이 없어야 한다.

사용:
  /workspace/.venv/bin/python3 loop/scripts/s0_train_detector.py
  python loop/scripts/s0_train_detector.py --tiny --limit 8 --device cpu   # 스모크
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import data as D
from loop_lib import detector as DET
from loop_lib import io_utils as io
from loop_lib import metrics as M


def train_full(name, model_name, pairs_by, cfg, device, out_dir, tiny, epochs=None):
    """train 전체로 학습 → dev-cal 로 에폭 선택 → τ 캘리브레이션 → 저장."""
    tr_texts, tr_labels, tr_ids = D.texts_labels(pairs_by["train"])
    cal, _gate = D.dev_halves(pairs_by["dev"])
    cal_texts, cal_labels, _ = D.texts_labels(cal)
    D.assert_no_test_docs(tr_ids, f"stage0/{name}", test_ids={r["doc_id"] for r in pairs_by["test"]})

    factory = DET.make_factory(model_name, tiny=tiny)
    model, tok, hist = DET.train_detector(
        factory, tr_texts, tr_labels, cfg.detector, device,
        dev_texts=cal_texts, dev_labels=cal_labels, epochs=epochs, log_prefix=name,
    )
    cal_h = DET.predict_scores(model, tok, [r["human_text"] for r in cal], device, max_len=cfg.detector.max_len)
    tau = DET.calibrate_tau(cal_h, cfg.detector.fpr_target)
    DET.save_detector(model, tok, out_dir, {"base_model": model_name, "history": hist,
                                            "train_pairs": len(pairs_by["train"]), "tiny": tiny})
    io.write_json(Path(out_dir) / "tau.json",
                  {"tau": tau, "fpr_target": cfg.detector.fpr_target,
                   "calibrated_on": "dev_cal_human", "n_human": len(cal)})
    print(f"[{name}] τ={tau:.4f} (dev-cal 인간 {len(cal)}건, FPR {cfg.detector.fpr_target:.0%})")
    return model, tok, tau


def score_pairs(model, tok, pairs, device, max_len):
    h = DET.predict_scores(model, tok, [r["human_text"] for r in pairs], device, max_len=max_len)
    a = DET.predict_scores(model, tok, [r["ai_text"] for r in pairs], device, max_len=max_len)
    return h, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="split 별 문서 수 제한 (스모크)")
    ap.add_argument("--tiny", action="store_true", help="초소형 랜덤 모델 (스모크)")
    ap.add_argument("--kfold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--skip-anchor", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    io.set_seed(cfg.seed)
    device = io.pick_device(a.device)
    out = C.stage0_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out}")

    pairs = D.load_pairs(cfg, limit=a.limit)
    by = D.by_split(pairs)
    cal, gate = D.dev_halves(by["dev"])
    print(f"pairs: train {len(by['train'])} / dev {len(by['dev'])} (cal {len(cal)} + gate {len(gate)}) / test {len(by['test'])}")

    # ── ① k-fold OOF (train 전용 — Stage 1 필터의 유일한 점수원) ──
    oof_path = out / "oof_scores.jsonl.gz"
    if a.skip_oof:
        print("OOF 생략 (--skip-oof)")
    elif oof_path.exists() and not a.force:
        print(f"OOF 있음 — 재사용: {oof_path}")
    else:
        k = a.kfold or cfg.detector.kfold
        factory = DET.make_factory(cfg.detector.model, tiny=a.tiny)
        oof = DET.kfold_oof(by["train"], factory, cfg.detector, device, k=k, seed=cfg.seed)
        io.write_jsonl(oof_path, [{"doc_id": d, **v} for d, v in oof.items()])
        print(f"OOF {len(oof)}건 → {oof_path}")

    # ── ② 최종 D₀ ────────────────────────────────────────────────
    d0_dir = out / "detector_d0"
    if (d0_dir / "tau.json").exists() and not a.force:
        print(f"D₀ 있음 — 재사용: {d0_dir}")
        model, tok = DET.load_detector(d0_dir, device)
        tau = io.read_json(d0_dir / "tau.json")["tau"]
    else:
        model, tok, tau = train_full("d0", cfg.detector.model, by, cfg, device, d0_dir,
                                     a.tiny, epochs=a.epochs)

    # ── ③ dev/test 점수 (동결 기준선) ────────────────────────────
    ml = cfg.detector.max_len
    dev_h, dev_a = score_pairs(model, tok, by["dev"], device, ml)
    test_h, test_a = score_pairs(model, tok, by["test"], device, ml)
    io.write_jsonl(out / "scores_dev.jsonl.gz",
                   [{"doc_id": r["doc_id"], "half": D.dev_half(r["doc_id"]),
                     "human": float(h), "ai": float(x)}
                    for r, h, x in zip(by["dev"], dev_h, dev_a)])
    io.write_jsonl(out / "scores_test.jsonl.gz",
                   [{"doc_id": r["doc_id"], "human": float(h), "ai": float(x)}
                    for r, h, x in zip(by["test"], test_h, test_a)])

    import numpy as np
    dev_auc = DET.auc_of([0] * len(dev_h) + [1] * len(dev_a), np.concatenate([dev_h, dev_a]))
    test_auc = DET.auc_of([0] * len(test_h) + [1] * len(test_a), np.concatenate([test_h, test_a]))

    # ── ④ 동결 앵커 2호 (KoELECTRA) ──────────────────────────────
    anchor_metrics = {}
    if not a.skip_anchor:
        an_dir = out / "anchor_koelectra"
        if (an_dir / "tau.json").exists() and not a.force:
            print(f"앵커 있음 — 재사용: {an_dir}")
        else:
            am, atok, atau = train_full("anchor", cfg.anchor.model, by, cfg, device, an_dir,
                                        a.tiny, epochs=a.epochs)
            ah, aa = score_pairs(am, atok, by["test"], device, ml)
            anchor_metrics = {
                "test_auc": DET.auc_of([0] * len(ah) + [1] * len(aa), np.concatenate([ah, aa])),
                "test_tpr_at_tau": DET.tpr_at(aa, atau),
                "test_fpr_at_tau": DET.fpr_at(ah, atau),
            }
            del am

    # ── ⑤ 하한선·안전장치 프로브 ─────────────────────────────────
    tr_h = [r["human_text"] for r in by["train"]]
    tr_a = [r["ai_text"] for r in by["train"]]
    te_h = [r["human_text"] for r in by["test"]]
    te_a = [r["ai_text"] for r in by["test"]]
    linear = M.linear_probe_transfer(tr_h, tr_a, te_h, te_a)
    len_auc = M.length_auc(te_h, te_a)

    report = {
        "n_pairs": {k: len(v) for k, v in by.items()},
        "tau0": tau,
        "d0": {
            "dev_auc": dev_auc,
            "test_auc": test_auc,
            "test_tpr_at_tau": DET.tpr_at(test_a, tau),
            "test_fpr_at_tau": DET.fpr_at(test_h, tau),
            "gate_half_fpr_at_tau": DET.fpr_at(
                [s for r, s in zip(by["dev"], dev_h) if D.dev_half(r["doc_id"]) == "gate"], tau),
        },
        "anchor_koelectra": anchor_metrics,
        "baselines": {
            "linear_probe_test_auc": linear,
            "length_only_test_auc": len_auc,
            "reference": "선형 0.9932 · 길이 0.5153 (notes/31 실측, 2026-08-13)",
        },
    }
    io.write_json(out / "report.json", report)
    md = [
        "# Stage 0 리포트 — 대리 탐지기 D₀",
        "",
        f"- pair: train {len(by['train'])} / dev {len(by['dev'])} / test {len(by['test'])}",
        f"- **τ₀ = {tau:.4f}** (dev-cal 인간, FPR {cfg.detector.fpr_target:.0%})",
        f"- D₀ dev AUC **{dev_auc:.4f}** · test AUC **{test_auc:.4f}**",
        f"- D₀ test TPR@τ **{report['d0']['test_tpr_at_tau']:.4f}** · test 인간 FPR@τ {report['d0']['test_fpr_at_tau']:.4f}",
        f"- 선형 베이스라인 test AUC {linear:.4f} (하한선 — D₀ 가 이보다 낮으면 학습 실패)",
        f"- 길이 단독 test AUC {len_auc:.4f} (0.5 근처여야 함)",
    ]
    if anchor_metrics:
        md.append(f"- KoELECTRA 앵커 test AUC {anchor_metrics['test_auc']:.4f} · TPR@τ {anchor_metrics['test_tpr_at_tau']:.4f}")
    io.write_text(out / "report.md", "\n".join(md) + "\n")
    print("\n".join(md))

    if not a.tiny and linear == linear and test_auc < linear:  # NaN 가드
        print("⚠ D₀ test AUC 가 선형 베이스라인보다 낮다 — notes/31 하한선 위반, 학습 설정 점검 필요")


if __name__ == "__main__":
    main()
