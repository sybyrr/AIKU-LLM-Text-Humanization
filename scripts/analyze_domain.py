#!/usr/bin/env python3
"""도메인 분석: 4텍스트셋(human / x_ai / SFT / DPO)에 대해
  - 도메인 roberta-D 의 CLS 특징 → t-SNE 2D 산점도 (PNG)
  - roberta-D P(AI) 평균
  - SCRN(news 학습, 전이/독립 탐지기) P(AI) 평균
Okabe-Ito 색맹안전 팔레트. 라벨은 영문(matplotlib 한글폰트 회피)."""
import matplotlib; matplotlib.use("Agg")
import argparse, json, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from stage2_sft import StyleBART, MODEL
from transformers.modeling_outputs import BaseModelOutput
DEV = "cuda:0"
PALETTE = {"Human": "#0072B2", "x_ai": "#D55E00", "SFT": "#009E73", "DPO": "#CC79A7"}


def load_bart(path):
    ck = torch.load(path, map_location=DEV)["model"]
    fs = tuple(ck["fusion.weight"].shape); kind = "concat" if fs[1] == 2*fs[0] else "additive"
    m = StyleBART().to(DEV)
    if kind == "additive":
        d = m.bart.config.d_model; m.fusion = nn.Linear(d, d).to(DEV)
    m.load_state_dict(ck); m.eval(); return m


@torch.no_grad()
def gen_all(m, tok, xs, bs=24):
    outs = []
    for i in range(0, len(xs), bs):
        e = tok(xs[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        cr = m.encode(e.input_ids, e.attention_mask)
        fw = m.fusion.weight
        if fw.shape[1] == 2*fw.shape[0]:
            sb = m.hsr.view(1,1,-1).expand(cr.size(0), cr.size(1), -1); fused = m.fusion(torch.cat([cr, sb], -1))
        else:
            fused = cr + m.fusion(cr + m.hsr)
        o = m.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                            attention_mask=e.attention_mask, num_beams=4, max_length=1024)
        outs.extend(tok.decode(x, skip_special_tokens=True) for x in o)
    return outs


@torch.no_grad()
def rob_feat_score(rob, rtok, texts, bs=32):
    feats, pai = [], []
    for i in range(0, len(texts), bs):
        b = rtok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        out = rob(**b, output_hidden_states=True)
        feats.append(out.hidden_states[-1][:, 0].float().cpu())
        pai.extend(torch.softmax(out.logits, -1)[:, 1].cpu().tolist())
    return torch.cat(feats).numpy(), pai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--sft", required=True); ap.add_argument("--dpo", required=True)
    ap.add_argument("--roberta", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrn", default="/workspace/models/news_scrn/scrn_best.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(args.pairs) if json.loads(l).get("split") == "test"][:args.n]
    human = [r["human_text"] for r in rows]; xai = [r["ai_text"] for r in rows]
    print(f"[{args.domain}] n={len(rows)} · SFT/DPO 생성...", flush=True)
    sft = load_bart(args.sft); sft_out = gen_all(sft, tok, xai); del sft; torch.cuda.empty_cache()
    dpo = load_bart(args.dpo); dpo_out = gen_all(dpo, tok, xai); del dpo; torch.cuda.empty_cache()

    sets = {"Human": human, "x_ai": xai, "SFT": sft_out, "DPO": dpo_out}
    rtok = AutoTokenizer.from_pretrained(args.roberta)
    rob = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(DEV).eval()
    feats, dscore = {}, {}
    for k, txt in sets.items():
        feats[k], p = rob_feat_score(rob, rtok, txt); dscore[k] = float(np.mean(p))
    del rob; torch.cuda.empty_cache()

    # SCRN (news 학습 전이 탐지기)
    sscore = {}
    try:
        from scrn_train import SCRN, BACKBONE
        from scrn_score import score_batch
        stok = AutoTokenizer.from_pretrained(BACKBONE)
        sm = SCRN().to(DEV); sm.load_state_dict(torch.load(args.scrn, map_location=DEV)["model"] if "model" in torch.load(args.scrn, map_location=DEV) else torch.load(args.scrn, map_location=DEV)); sm.eval()
        for k, txt in sets.items():
            ps, _ = score_batch(sm, stok, txt); sscore[k] = float(np.mean(ps))
    except Exception as e:
        print("SCRN 스킵:", str(e)[:150], flush=True)
        sscore = {k: None for k in sets}

    # t-SNE
    X = np.concatenate([feats[k] for k in sets]); lab = sum(([k]*len(sets[k]) for k in sets), [])
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    plt.figure(figsize=(8, 7))
    off = 0
    for k in sets:
        n = len(sets[k]); e = emb[off:off+n]; off += n
        plt.scatter(e[:, 0], e[:, 1], s=42, c=PALETTE[k], edgecolors="white", linewidths=0.6,
                    alpha=0.85, label=f"{k}  D={dscore[k]:.2f}" + (f" S={sscore[k]:.2f}" if sscore[k] is not None else ""))
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.title(f"{args.domain}: roberta-D CLS features (t-SNE)\nD=roberta P(AI)  S=SCRN P(AI)  ·  n={args.n}/set", fontsize=11)
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\n[{args.domain}] 저장: {args.out}")
    print("  roberta-D P(AI):", {k: round(v, 3) for k, v in dscore.items()})
    print("  SCRN     P(AI):", {k: (round(v, 3) if v is not None else None) for k, v in sscore.items()})


if __name__ == "__main__":
    main()
