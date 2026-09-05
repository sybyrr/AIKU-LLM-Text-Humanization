#!/usr/bin/env python3
"""SFT vs DPO 비교 리뷰 HTML — 원문(x_ai) / SFT 생성 / DPO 생성 / 사람참조 4단.
DPO가 SFT 대비 무엇을 바꿨는지(또는 거의 안 바꿨는지) 눈으로 확인. plain beam4."""
import argparse, json, html, difflib, datetime
import torch, torch.nn as nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from stage2_sft import StyleBART, MODEL
DEV = "cuda:0"


def load_any(path):
    ck = torch.load(path, map_location=DEV)["model"]
    fs = tuple(ck["fusion.weight"].shape)
    kind = "concat" if fs[1] == 2 * fs[0] else "additive"
    m = StyleBART().to(DEV)
    if kind == "additive":
        d = m.bart.config.d_model
        m.fusion = nn.Linear(d, d).to(DEV)
    m.load_state_dict(ck); m.eval()
    return m


def fuse(m, cr, style):
    fw = m.fusion.weight
    if fw.shape[1] == 2 * fw.shape[0]:
        sb = style.view(1, 1, -1).expand(cr.size(0), cr.size(1), -1)
        return m.fusion(torch.cat([cr, sb], dim=-1))
    return cr + m.fusion(cr + style)


@torch.no_grad()
def gen(m, tok, x):
    e = tok(x, return_tensors="pt", truncation=True, max_length=512).to(DEV)
    cr = m.encode(e.input_ids, e.attention_mask)
    fused = fuse(m, cr, m.hsr)
    o = m.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        attention_mask=e.attention_mask, num_beams=4, max_length=1024)
    return tok.decode(o[0], skip_special_tokens=True)


def esc(t):
    return html.escape(t or "").replace("\n", "<br>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="dpo")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(args.pairs)
            if json.loads(l).get("split") == args.split][:args.n]
    sft = load_any(args.sft)
    dpo = load_any(args.dpo)

    cards, n_same, sims = [], 0, []
    for r in rows:
        gs = gen(sft, tok, r["ai_text"])
        gd = gen(dpo, tok, r["ai_text"])
        sim = difflib.SequenceMatcher(None, gs, gd).ratio()
        sims.append(sim)
        same = sim > 0.98
        if same:
            n_same += 1
        badge = f'<span class="b {"same" if same else "diff"}">SFT≈DPO {sim:.0%}</span>'
        cards.append(f"""
<div class="card">
  <div class="hd"><span class="doc">{esc(r.get('id',''))}</span>{badge}</div>
  <div class="cols">
    <div class="col"><div class="lbl">인간 원문 x_human</div><div class="txt hu">{esc(r['human_text'])}</div></div>
    <div class="col"><div class="lbl">x_ai (AI 재서술)</div><div class="txt ai">{esc(r['ai_text'])}</div></div>
    <div class="col"><div class="lbl">SFT 생성</div><div class="txt sft">{esc(gs)}</div></div>
    <div class="col"><div class="lbl">DPO 생성</div><div class="txt dpo">{esc(gd)}</div></div>
  </div>
</div>""")

    mean = lambda x: sum(x) / len(x) if x else 0
    date = datetime.date.today().isoformat()
    summary = f"n={len(rows)} · SFT-DPO 평균 유사도 {mean(sims):.0%} · 거의 동일(≥98%) {n_same}/{len(rows)}건"
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SFT vs DPO {args.tag} {date}</title>
<style>
:root{{--bg:#0f1115;--card:#1a1d24;--bd:#2a2f3a;--tx:#e6e8ec;--mut:#9aa4b2;--ai:#8ab4f8;--sft:#7ee787;--dpo:#c792ea;--hu:#f0b37e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:13px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif}}
header{{position:sticky;top:0;background:#0f1115ee;backdrop-filter:blur(6px);padding:14px 22px;border-bottom:1px solid var(--bd);z-index:9}}
header h1{{margin:0 0 4px;font-size:17px}} header .sum{{color:var(--mut);font-size:13px}}
.wrap{{padding:18px 22px;max-width:1800px;margin:0 auto}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin-bottom:14px;overflow:hidden}}
.hd{{display:flex;justify-content:space-between;align-items:center;padding:9px 13px;border-bottom:1px solid var(--bd);background:#141821}}
.doc{{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace}}
.b{{padding:2px 8px;border-radius:6px;font-size:12px}} .b.same{{background:#1f3326;color:var(--sft)}} .b.diff{{background:#2e2440;color:var(--dpo)}}
.cols{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0}}
.col{{padding:11px 13px;border-right:1px solid var(--bd)}} .col:last-child{{border-right:none}}
.lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin-bottom:6px}}
.txt{{white-space:normal;word-break:break-word}}
.txt.ai{{border-left:3px solid var(--ai);padding-left:9px}} .txt.sft{{border-left:3px solid var(--sft);padding-left:9px}}
.txt.dpo{{border-left:3px solid var(--dpo);padding-left:9px}} .txt.hu{{border-left:3px solid var(--hu);padding-left:9px}}
@media(max-width:1100px){{.cols{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<header><h1>SFT vs DPO 비교 · {esc(args.tag)}</h1><div class="sum">{date} · plain beam4 · {summary}<br>
SFT=<code>{esc(args.sft)}</code> · DPO=<code>{esc(args.dpo)}</code></div></header>
<div class="wrap">{''.join(cards)}</div></body></html>"""
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(page)
    print(f"저장: {args.out}\n요약: {summary}")


if __name__ == "__main__":
    main()
