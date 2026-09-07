#!/usr/bin/env python3
"""CPU 스모크 — tiny 랜덤 모델로 전 파이프라인이 끝까지 도는지만 확인한다.

수렴은 검증하지 않는다(서버 몫). 여기서 잡는 것은 코드 버그: shape 불일치, 잘못된 키,
경로 버그, 단계 간 조인 실패. 실데이터를 split 당 소량만 쓰고, 모델은 tiny=True 랜덤.

전제: klue/roberta-base · gogamza/kobart-base-v2 · monologg/koelectra 토크나이저가
캐시돼 있어야 한다 (최초 1회 인터넷 필요).

사용:  python loop/smoke/run_smoke.py [--limit 12]
종료코드 0 = 전 단계 통과.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "loop"
SC = LOOP / "scripts"
sys.path.insert(0, str(LOOP))
from loop_lib import config as C  # noqa: E402
from loop_lib import io_utils as io  # noqa: E402


def make_smoke_config(run_dir):
    """base.yaml + 스모크 오버라이드 → 임시 full config. 반환: 파일 경로."""
    import yaml

    base = yaml.safe_load((LOOP / "configs" / "base.yaml").read_text(encoding="utf-8"))
    ov = {
        "paths": {"runs": str(run_dir).replace("\\", "/")},
        "detector": {"batch_size": 4, "grad_accum": 1, "epochs": 1, "kfold": 2, "max_len": 128},
        "sft": {"batch_size": 4, "epochs": 2, "patience": 5, "warmup_ratio": 0.1},
        "paraphraser": {"max_src_len": 128, "max_tgt_len": 128},
        "rounds": {"total": 2, "n_candidates": 4, "n_candidates_max": 6, "gen_batch_size": 2,
                   "max_pairs_per_doc": 2},
        "dpo": {"batch_size": 2, "grad_accum": 1, "epochs": 1},
        # greedy=False: tiny 랜덤 모델은 greedy argmax 가 한 토큰만 반복해 빈 문자열이 된다.
        # 실모델은 greedy(결정적)가 맞다 — 스모크에서만 샘플링으로 비어있지 않게 한다.
        "eval": {"eval_batch_size": 4, "enable_simcse": False, "enable_ppl": False,
                 "ck_batch_size": 8, "greedy": False},
    }
    from loop_lib.config import _merge

    merged = _merge(base, ov)
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "smoke_config.yaml"
    p.write_text(yaml.safe_dump(merged, allow_unicode=True), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--keep", action="store_true", help="_run 폴더 보존 (기본은 지우고 시작)")
    a = ap.parse_args()

    run_dir = LOOP / "smoke" / "_run"
    if run_dir.exists() and not a.keep:
        import shutil

        shutil.rmtree(run_dir)
    cfg_path = make_smoke_config(run_dir)
    arm = str(LOOP / "configs" / "arms" / "main.yaml")
    L = str(a.limit)
    common = ["--config", str(cfg_path), "--device", "cpu"]

    steps = [
        ("s0 탐지기+앵커+OOF", [SC / "s0_train_detector.py", *common, "--tiny", "--limit", L]),
        ("s1 D_pair 필터", [SC / "s1_build_dpair.py", "--config", str(cfg_path), "--limit", L,
                            "--keep-min", "6"]),
        ("s2 SFT", [SC / "s2_sft.py", *common, "--tiny", "--limit", L]),
        ("r1 후보생성", [SC / "r_generate.py", *common, "--arm-config", arm, "--round", "1", "--limit", L]),
        ("r1 prefs", [SC / "r_build_prefs.py", *common, "--arm-config", arm, "--round", "1",
                      "--limit", L, "--fallback-top1", "--force"]),
        ("r1 DPO", [SC / "r_dpo.py", *common, "--arm-config", arm, "--round", "1"]),
        ("r1 평가", [SC / "r_eval.py", *common, "--arm-config", arm, "--round", "1", "--limit", L, "--light"]),
        ("r1 탐지기재학습+게이트", [SC / "r_retrain_detector.py", *common, "--arm-config", arm,
                              "--round", "1", "--tiny", "--limit", L]),
        # 라운드 체이닝: t=2 는 G_1(round1/dpo/final) 과 D_2(round1/detector/model) 를 읽어야 한다
        ("r2 후보생성(체이닝)", [SC / "r_generate.py", *common, "--arm-config", arm, "--round", "2", "--limit", L]),
        ("r2 prefs(체이닝)", [SC / "r_build_prefs.py", *common, "--arm-config", arm, "--round", "2",
                            "--limit", L, "--fallback-top1", "--force"]),
        ("r2 DPO(체이닝)", [SC / "r_dpo.py", *common, "--arm-config", arm, "--round", "2"]),
    ]

    print(f"스모크 시작 — limit={a.limit} run={run_dir}\n")
    for i, (label, cmd) in enumerate(steps, 1):
        cmd = [sys.executable] + [str(x) for x in cmd]
        t = time.time()
        print(f"── [{i}/{len(steps)}] {label}")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n✗ 실패: {label}  (exit {r.returncode})\n   {' '.join(cmd)}")
            sys.exit(1)
        print(f"   ✓ {time.time()-t:.1f}s\n")

    # ── 산출물 존재·형태 검증 ──────────────────────────────────
    cfg = C.load_config(cfg_path, arm)
    checks = [
        C.stage0_dir(cfg) / "detector_d0" / "tau.json",
        C.stage0_dir(cfg) / "anchor_koelectra" / "tau.json",
        C.stage0_dir(cfg) / "oof_scores.jsonl.gz",
        C.stage1_dpair(cfg),
        C.stage2_dir(cfg) / "best" / "style_head.pt",
        C.round_dir(cfg, 1) / "dpo" / "final" / "style_head.pt",
        C.round_dir(cfg, 1) / "eval" / "metrics.json",
        C.round_dir(cfg, 1) / "detector" / "gate.json",
        C.round_dir(cfg, 2) / "dpo" / "final" / "style_head.pt",
    ]
    print("── 산출물 검증")
    bad = [p for p in checks if not Path(p).exists()]
    for p in checks:
        print(f"   {'✓' if Path(p).exists() else '✗'} {Path(p).relative_to(ROOT)}")
    if bad:
        print(f"\n✗ 누락 {len(bad)}건")
        sys.exit(1)

    # 게이트·평가 스키마 sanity
    gate = io.read_json(C.round_dir(cfg, 1) / "detector" / "gate.json")
    metrics = io.read_json(C.round_dir(cfg, 1) / "eval" / "metrics.json")
    assert "collapsed" in gate and "frozen_d0" in metrics["detectors"], "산출물 스키마 이상"
    print(f"   gate.collapsed={gate['collapsed']} · frozen_d0.gen_asr={metrics['detectors']['frozen_d0']['gen_asr']:.3f}")

    # 다른 arm 설정·replay 경로 sanity (full 외 latest_only 도 조립되는지).
    # latest_only arm 은 실제로 돌리지 않았으므로, main 의 round1 replay_gen 을 그 경로로
    # 복사해 build_buffer 의 분기 로직만 검증한다.
    import shutil

    from loop_lib import data as Dm
    from loop_lib import replay as Rm

    cfg_lo = C.load_config(cfg_path, str(LOOP / "configs" / "arms" / "latest_only.yaml"))
    src = Rm.replay_gen_path(cfg, 1)                 # main arm (cfg.arm.name=main)
    dst = Rm.replay_gen_path(cfg_lo, 1)              # latest_only arm
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    pairs = Dm.by_split(Dm.load_pairs(cfg_lo, limit=a.limit))["train"]
    texts, labels, weights, info = Rm.build_buffer(cfg_lo, pairs, t=1)
    assert len(texts) == len(labels) == len(weights) and info["mode"] == "latest_only"
    assert "orig_ai" not in info, "latest_only 인데 원본 AI 가 버퍼에 들어갔다"
    print(f"   latest_only replay buffer OK: {info}")

    print("\n✓ 스모크 전 단계 통과")


if __name__ == "__main__":
    main()
