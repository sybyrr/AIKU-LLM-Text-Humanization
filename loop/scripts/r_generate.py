#!/usr/bin/env python3
"""Round 단계 ① — G_{t-1} 에서 원문당 후보 n=8 temperature 샘플링 (빔서치 아님).

- 대상: D_pair 의 train 문서 (x_ai 를 입력으로 s_human 스타일 재서술)
- resume: candidates 파일에 이미 있는 문서는 건너뛴다 (배치 단위 append — 중단 안전)
- --top-up: prefs_report.json 의 needs_more 문서만 n_candidates 만큼 증량
  (hard negative 부족 시 τ를 낮추지 말고 후보를 늘린다 — notes/31 Round ②,
   문서당 상한 n_candidates_max)

사용:
  python loop/scripts/r_generate.py --arm-config loop/configs/arms/main.yaml --round 1
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import io_utils as io
from loop_lib import paraphraser as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--arm-config", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-up", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config, a.arm_config)
    t = a.round
    io.set_seed(cfg.seed + t * 1000)
    device = io.pick_device(a.device)
    rdir = C.round_dir(cfg, t)
    out_path = rdir / "candidates.jsonl.gz"
    n_target = cfg.rounds.n_candidates

    rows = [r for r in io.read_jsonl(C.stage1_dpair(cfg)) if r["split"] == "train"]
    if a.limit:
        rows = rows[: a.limit]
    by_doc = {r["doc_id"]: r for r in rows}

    have = collections.Counter()
    if out_path.exists():
        for r in io.iter_jsonl(out_path):
            have[r["doc_id"]] += 1

    if a.top_up:
        rep_path = rdir / "prefs_report.json"
        needs = io.read_json(rep_path).get("needs_more", []) if rep_path.exists() else []
        todo = [(d, min(have[d] + n_target, cfg.rounds.n_candidates_max) - have[d])
                for d in needs if d in by_doc]
        todo = [(d, k) for d, k in todo if k > 0]
    else:
        todo = [(d, n_target - have[d]) for d in by_doc if have[d] < n_target]

    if not todo:
        print(f"할 일 없음 — 후보 충분 ({out_path})")
        return

    gen_dir = C.generator_in(cfg, t)
    print(f"G_{t-1} = {gen_dir}  대상 {len(todo)}문서  device={device}")
    model, tok = P.load(gen_dir, device)

    bs = cfg.rounds.gen_batch_size
    done = 0
    for s in range(0, len(todo), bs):
        chunk = todo[s : s + bs]
        k = max(x[1] for x in chunk)  # 배치 내 최대 필요 수만큼 뽑고 문서별로 자른다
        texts = [by_doc[d]["ai_text"] for d, _ in chunk]
        outs = P.paraphrase_texts(model, tok, texts, device, cfg.paraphraser,
                                  style=P.STYLE_HUMAN, num_return=k, greedy=False,
                                  temperature=cfg.rounds.gen_temperature,
                                  top_p=cfg.rounds.gen_top_p, batch_size=bs,
                                  min_new_tokens=cfg.paraphraser.get("min_new_tokens", 0))
        new_rows = []
        for (d, need), cands in zip(chunk, outs):
            base_idx = have[d]
            written = 0
            for text in cands[:need]:
                if text:
                    new_rows.append({"doc_id": d, "idx": base_idx + written, "text": text})
                    written += 1
            have[d] += written
        io.append_jsonl(out_path, new_rows)
        done += len(chunk)
        if done % (bs * 20) < bs:
            print(f"  {done}/{len(todo)} 문서", flush=True)

    total = sum(have.values())
    print(f"후보 총 {total}건 (문서 {len(have)}개) → {out_path}")


if __name__ == "__main__":
    main()
