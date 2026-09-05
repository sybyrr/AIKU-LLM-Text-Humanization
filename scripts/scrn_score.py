#!/usr/bin/env python3
"""② 전이세트 채점 — 학습된 SCRN 으로 humanizer 출력을 채점(detect_binoculars.py 관례 미러링).

입력: transfer_human.jsonl(body=인간기준) + transfer_gen.jsonl(doc_id/text/model/cond).
출력: {doc_id,label,model,cond,score,n_tok}  (score=P(AI), 클수록 AI; human행 append)
      → aggregate_transfer.py 의 (name,path) 리스트에 그대로 물림.
사용: python scrn_score.py --gen ../dataset/news_track/transfer_gen.jsonl \
        --out ../dataset/news_track/transfer_scrn.jsonl
"""
import argparse, json, torch
from transformers import AutoTokenizer
from scrn_train import SCRN, BACKBONE, DEV, NT


@torch.no_grad()
def score_batch(model, tok, texts, max_len=512, bs=32):
    ps, ntok = [], []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        b = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        ps.extend(model.score(b).float().cpu().tolist())
        ntok.extend(b.attention_mask.sum(1).cpu().tolist())
    return ps, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=f"{NT}/transfer_human.jsonl")
    ap.add_argument("--gen", default=f"{NT}/transfer_gen.jsonl")
    ap.add_argument("--out", default=f"{NT}/transfer_scrn.jsonl")
    ap.add_argument("--best", default="/workspace/models/news_scrn/scrn_best.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    st = torch.load(args.best, map_location=DEV)
    model = SCRN(backbone=st.get("backbone", BACKBONE), dz=st.get("dz", 512)).to(DEV).eval()
    model.load_state_dict(st["model"])
    print(f"로드: {args.best} (dev AUROC {st.get('dev_auroc'):.4f})", flush=True)

    rows = []
    # 인간 기준
    H = [json.loads(l) for l in open(args.human) if l.strip()]
    ps, nt = score_batch(model, tok, [r["body"] for r in H])
    for r, p, n in zip(H, ps, nt):
        rows.append({"doc_id": r["doc_id"], "label": "human", "model": "human",
                     "cond": None, "score": round(p, 6), "n_tok": int(n)})
    print(f"인간 {len(H)}편", flush=True)

    # 생성물 (조건별)
    G = [json.loads(l) for l in open(args.gen) if l.strip() and json.loads(l).get("text")]
    ps, nt = score_batch(model, tok, [g["text"] for g in G])
    for g, p, n in zip(G, ps, nt):
        rows.append({"doc_id": g["doc_id"], "label": "ai", "model": g["model"],
                     "cond": g["cond"], "score": round(p, 6), "n_tok": int(n)})

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 집계 — 조건별 평균 P(AI) + 인간 95%분위 이하(=속임) 비율
    import numpy as np
    hs = np.array([r["score"] for r in rows if r["label"] == "human"])
    thr = float(np.percentile(hs, 95))
    print(f"\n인간 P(AI): 중앙 {np.median(hs):.3f} · 95%분위 {thr:.3f} (속임 임계)")
    by = {}
    for r in rows:
        if r["label"] == "ai":
            by.setdefault(r["model"], []).append(r["score"])
    print(f"{'조건':<10}{'n':>5}{'평균P(AI)':>12}{'속임율%':>10}")
    for m in sorted(by):
        v = np.array(by[m])
        fooled = 100 * float(np.mean(v <= thr))
        print(f"{m:<10}{len(v):>5}{v.mean():>12.3f}{fooled:>10.1f}")
    print(f"\n저장: {args.out}  ({len(rows)}행)")


if __name__ == "__main__":
    main()
