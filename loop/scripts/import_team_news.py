#!/usr/bin/env python3
"""팀 뉴스 D2 산출물을 사용자 루프 형식으로 스냅샷/변환한다.

원본 팀 디렉터리는 읽기만 한다. 데이터는 gzip JSONL 정본으로 복사하고,
StyleBART 단일 ``.pt`` 체크포인트를 루프의 HF base + style_head 형식으로 변환한다.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from loop_lib import config as C
from loop_lib import io_utils as io
from loop_lib import paraphraser as P


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_generator(src, dst, cfg):
    ck = torch.load(src, map_location="cpu", weights_only=True)
    state = ck["model"]
    model, tok = P.create(cfg.paraphraser.model)
    base_state = {k[len("bart."):]: v for k, v in state.items() if k.startswith("bart.")}
    incompatible = model.base.load_state_dict(base_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"BART 변환 키 불일치: {incompatible}")
    with torch.no_grad():
        model.style_emb.weight[0].copy_(state["asr"])
        model.style_emb.weight[1].copy_(state["hsr"])
    model.fusion.load_state_dict({"weight": state["fusion.weight"], "bias": state["fusion.bias"]})
    P.save(model, tok, dst, {
        "source": str(src),
        "source_sha256": sha256(src),
        "source_config": ck.get("config", {}),
        "conversion": "team StyleBART(asr/hsr/bart) -> loop StyleParaphraser",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=str(C.LOOP_DIR / "configs" / "arms" / "team_news.yaml"))
    ap.add_argument("--team-root", required=True)
    ap.add_argument("--tau", type=float, default=0.062,
                    help="팀 D2 held-out human FPR 5%% 임계값(로그 보고값)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    team = Path(a.team_root).resolve()
    src_data = team / "dataset/news_track/news_dpair_D2.jsonl"
    src_gen = team / "models/news_dpo_D2/dpo_bart.pt"
    src_det = team / "models/news_roberta_D2"
    for p in (src_data, src_gen, src_det):
        if not p.exists():
            raise FileNotFoundError(p)

    snapshot = C.rp(cfg.data.paired_jsonl)
    stage1 = C.stage1_dpair(cfg)
    gen_dst = C.generator_in(cfg, 1)
    d0_dst = C.stage0_dir(cfg) / "detector_d0"
    manifest_path = C.runs_dir(cfg) / "import_manifest.json"
    if manifest_path.exists() and not a.force:
        print(f"가져오기 산출물 있음 — 재사용: {manifest_path}")
        return

    rows = []
    for r in io.iter_jsonl(src_data):
        rows.append({
            **r,
            "doc_id": r["id"],
            "kci_article_id": r["id"],
            "title": r.get("title", ""),
            "field": r.get("field", "news"),
        })
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    if len({r["doc_id"] for r in rows}) != len(rows):
        raise RuntimeError("팀 뉴스 D2 데이터에 중복 id가 있다")
    io.write_jsonl(snapshot, rows)
    io.write_jsonl(stage1, rows)

    if gen_dst.exists():
        shutil.rmtree(gen_dst)
    gen_dst.mkdir(parents=True, exist_ok=True)
    convert_generator(src_gen, gen_dst, cfg)

    if d0_dst.exists():
        shutil.rmtree(d0_dst)
    shutil.copytree(src_det, d0_dst)
    io.write_json(d0_dst / "tau.json", {
        "tau": a.tau,
        "fpr_target": cfg.detector.fpr_target,
        "calibrated_on": "team D2 held-out human (reported)",
        "source_log_metric": "FPR5 threshold 0.062",
    })

    # 독립 KoELECTRA 앵커는 기존 사용자 Stage0 산출물이 있으면 복제한다.
    anchor_src = C.REPO_ROOT / "loop/runs/stage0/anchor_koelectra"
    anchor_dst = C.stage0_dir(cfg) / "anchor_koelectra"
    if anchor_src.exists() and anchor_src.resolve() != anchor_dst.resolve():
        if anchor_dst.exists():
            shutil.rmtree(anchor_dst)
        shutil.copytree(anchor_src, anchor_dst)

    manifest = {
        "team_root": str(team),
        "sources": {
            "data": {"path": str(src_data), "sha256": sha256(src_data)},
            "generator": {"path": str(src_gen), "sha256": sha256(src_gen)},
            "detector": str(src_det),
        },
        "destinations": {
            "paired_snapshot": str(snapshot),
            "stage1_dpair": str(stage1),
            "initial_generator_g1": str(gen_dst),
            "frozen_detector_d0": str(d0_dst),
        },
        "rows": len(rows),
        "splits": counts,
        "tau_d0": a.tau,
    }
    io.write_json(manifest_path, manifest)
    print(f"팀 뉴스 D2 {len(rows):,}건 {counts} → {snapshot}")
    print(f"G1 변환 → {gen_dst}")
    print(f"동결 D0 복사(τ={a.tau}) → {d0_dst}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
