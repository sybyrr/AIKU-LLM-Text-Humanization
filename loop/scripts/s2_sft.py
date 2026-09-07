#!/usr/bin/env python3
"""Stage 2 — Style-injection SFT 실행 + 육안 검수 20건 덤프.

산출물 (loop/runs/stage2/):
  best/  last/     StyleParaphraser 체크포인트 (base/ + style_head.pt)
  history.json     에폭별 손실
  samples_dev.md   dev 20건: 입력 x_ai / 생성 / 정답 인간 — **ko-BART GO/NO-GO 판단 자료**
                   (notes/31: 실패 시 Qwen3-0.6B 전체 파인튜닝 fp32 로 대체)

사용:
  /workspace/.venv/bin/python3 loop/scripts/s2_sft.py
  python loop/scripts/s2_sft.py --tiny --limit 8 --epochs 2 --device cpu   # 스모크
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import io_utils as io
from loop_lib import paraphraser as P
from loop_lib import sft as S


def dump_samples(model, tok, rows, para_cfg, device, out_path, n=20):
    rows = rows[:n]
    gens = P.paraphrase_texts(model, tok, [r["ai_text"] for r in rows], device, para_cfg,
                              style=P.STYLE_HUMAN, greedy=True, batch_size=4)
    md = ["# SFT 육안 검수 — dev 20건 (x_ai → G(x_ai, s_human) vs 인간 원문)", "",
          "판단 기준(notes/31): 내용이 유지되고 문장이 한국어로 성립하는가. "
          "깨진 문장·내용 이탈이 다수면 ko-BART NO-GO → Qwen3-0.6B 대안 검토.", ""]
    for i, (r, g) in enumerate(zip(rows, gens), 1):
        md += [f"## {i}. {r['doc_id']}", "", f"**입력 (x_ai)**\n\n> {r['ai_text']}", "",
               f"**생성 (s_human)**\n\n> {g[0]}", "", f"**정답 (인간 원문)**\n\n> {r['human_text']}", ""]
    io.write_text(out_path, "\n".join(md))
    print(f"육안 검수 샘플 {len(rows)}건 → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="문서 수 제한 (스모크)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    io.set_seed(cfg.seed)
    device = io.pick_device(a.device)
    out = C.stage2_dir(cfg)

    if (out / "best" / "style_head.pt").exists() and not a.force:
        print(f"Stage 2 체크포인트 있음 — 재사용: {out} (--force 로 재학습)")
        model, tok = P.load(out / "best", device)
        rows = io.read_jsonl(C.stage1_dpair(cfg))
        dev = [r for r in rows if r["split"] == "dev"]
        dump_samples(model, tok, dev, cfg.paraphraser, device, out / "samples_dev.md")
        return

    rows = io.read_jsonl(C.stage1_dpair(cfg))
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if a.limit:
        train, dev = train[: a.limit], dev[: max(2, a.limit // 4)]
    assert train and dev, "D_pair 가 비어 있다 — s1_build_dpair.py 먼저"
    print(f"D_pair: train {len(train)} / dev {len(dev)}  device={device}")

    model, tok = P.create(cfg.paraphraser.model, tiny=a.tiny)
    S.run_sft(cfg, model, tok, train, dev, out, device, epochs=a.epochs)
    dump_samples(model, tok, dev, cfg.paraphraser, device, out / "samples_dev.md")
    print(f"완료 → {out}/best (다음: 육안 검수 후 r_eval.py --round 0 로 SFT 기준선 평가)")


if __name__ == "__main__":
    main()
