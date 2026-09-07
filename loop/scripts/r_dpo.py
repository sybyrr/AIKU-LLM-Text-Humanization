#!/usr/bin/env python3
"""Round 단계 ③ — DPO 학습. 정책 = G_{t-1}, 산출 = G_t (round{t}/dpo/final).

lr 5e-6 · β=0.1 · 5 epoch (notes/31). ref logp 는 prefs 에 사전계산돼 있다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import dpo as DPO
from loop_lib import io_utils as io
from loop_lib import paraphraser as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="pref 수 제한 (스모크)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    t = a.round
    io.set_seed(cfg.seed + t)
    device = io.pick_device(a.device)
    rdir = C.round_dir(cfg, t)
    out = rdir / "dpo"
    if (out / "final" / "style_head.pt").exists() and not a.force:
        print(f"DPO 체크포인트 있음 — 재사용: {out}/final (--force 로 재학습)")
        return

    prefs = io.read_jsonl(rdir / "prefs.jsonl.gz")
    if a.limit:
        prefs = prefs[: a.limit]
    if not prefs:
        raise SystemExit(f"prefs 0건 — r_build_prefs 산출 확인 ({rdir/'prefs.jsonl.gz'})")
    model, tok = P.load(C.generator_in(cfg, t), device)   # 정책 초기값 = G_{t-1}
    print(f"arm={cfg.arm.name} round={t} prefs={len(prefs)} device={device}")
    DPO.run_dpo(cfg, model, tok, prefs, out, device, epochs=a.epochs,
                log_prefix=f"dpo-{cfg.arm.name}-r{t}")
    print(f"G_{t} → {out}/final")


if __name__ == "__main__":
    main()
