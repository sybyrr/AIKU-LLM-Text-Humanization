#!/usr/bin/env python3
"""Stage2(EOS 수정) 결과 육안 리뷰용 HTML 생성.

held-out(clean20k test)에서 x_ai → 스타일 이식(HSR 경로) 생성 후, 원문/생성/사람참조를
나란히 보여준다. 각 샘플에 길이·seq-rep-3·붕괴여부 배지. 디코딩은 논문 세팅(beam=4).

사용:
  python make_stage2_review.py --ckpt /workspace/models/news_sft_eos20k/style_bart.pt \
     --n 30 --out /workspace/review/stage2_eos_review_YYYYMMDD.html
"""
import argparse, json, html, collections, datetime
import torch, torch.nn as nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from stage2_sft import StyleBART, MODEL

NT = "/workspace/dataset/news_track"; DEV = "cuda:0"


def load_any(path):
    ck = torch.load(path, map_location=DEV)["model"]
    fs = tuple(ck["fusion.weight"].shape)
    kind = "concat" if fs[1] == 2 * fs[0] else "additive"
    m = StyleBART().to(DEV)
    if kind == "additive":
        d = m.bart.config.d_model
        m.fusion = nn.Linear(d, d).to(DEV)
    m.load_state_dict(ck); m.eval()
    return m, kind


def fuse(m, cr, style, kind):
    if kind == "concat":
        sb = style.view(1, 1, -1).expand(cr.size(0), cr.size(1), -1)
        return m.fusion(torch.cat([cr, sb], dim=-1))
    return cr + m.fusion(cr + style)


@torch.no_grad()
def gen(m, kind, tok, x, dec):
    e = tok(x, return_tensors="pt", truncation=True, max_length=512).to(DEV)
    cr = m.encode(e.input_ids, e.attention_mask)
    fused = fuse(m, cr, m.hsr, kind)
    o = m.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        attention_mask=e.attention_mask, **dec)
    return tok.decode(o[0], skip_special_tokens=True)


def rep_n(toks, n=3):
    if len(toks) < n:
        return 0.0
    ng = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    return 1 - len(set(ng)) / len(ng)


def is_collapse(t):
    toks = t.split()
    return bool(toks) and max(collections.Counter(toks).values()) > 30


def esc(t):
    return html.escape(t).replace("\n", "<br>")


CARD = """
<div class="card">
  <div class="hd"><span class="doc">{doc}</span>
    <span class="badges">{badges}</span></div>
  <div class="cols">
    <div class="col"><div class="lbl">원문 (x_ai)</div><div class="txt ai">{ai}</div></div>
    <div class="col"><div class="lbl">EOS-SFT 생성 (x_human 스타일)</div><div class="txt gen">{gen}</div></div>
    <div class="col"><div class="lbl">사람 참조 (x_human)</div><div class="txt hu">{hu}</div></div>
  </div>
</div>"""


def badge(label, val, warn=False):
    cls = "b warn" if warn else "b"
    return f'<span class="{cls}">{label} {val}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", default=f"{NT}/news_dpair_clean20k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="eos20k")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(args.pairs)
            if json.loads(l).get("split") == args.split][:args.n]
    m, kind = load_any(args.ckpt)
    dec = dict(num_beams=4, max_length=1024)   # 논문 세팅: 맨 beam4 (반복방지 장치 없음)

    cards, lens, reps, colls, ratios = [], [], [], [], []
    for r in rows:
        g = gen(m, kind, tok, r["ai_text"], dec)
        gl = len(g.split()); reps3 = rep_n(g.split(), 3); coll = is_collapse(g)
        ratio = len(g) / max(len(r["ai_text"]), 1)
        lens.append(gl); reps.append(reps3); colls.append(coll); ratios.append(ratio)
        b = badge("어절", gl) + badge("len비", f"{ratio:.2f}", warn=ratio > 1.5) \
            + badge("rep3", f"{reps3:.2f}", warn=reps3 > 0.2) \
            + (badge("붕괴", "●", warn=True) if coll else "")
        cards.append(CARD.format(doc=esc(r["id"]), badges=b,
                                 ai=esc(r["ai_text"]), gen=esc(g), hu=esc(r["human_text"])))

    n = len(rows)
    mean = lambda x: sum(x) / len(x) if x else 0
    summary = (f"n={n} · 평균 어절 {mean(lens):.0f} · 평균 len비(생성/원문) {mean(ratios):.2f} · "
               f"평균 rep3 {mean(reps):.3f} · 붕괴율 {100*sum(colls)/n:.0f}%")
    date = datetime.date.today().isoformat()

    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage2 EOS 리뷰 {date}</title>
<style>
:root{{--bg:#0f1115;--card:#1a1d24;--bd:#2a2f3a;--tx:#e6e8ec;--mut:#9aa4b2;--ai:#8ab4f8;--gen:#7ee787;--hu:#f0b37e;--warn:#ff6b6b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font:14px/1.6 -apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif}}
header{{position:sticky;top:0;background:#0f1115ee;backdrop-filter:blur(6px);padding:16px 24px;border-bottom:1px solid var(--bd);z-index:9}}
header h1{{margin:0 0 4px;font-size:18px}}
header .sum{{color:var(--mut);font-size:13px}}
.wrap{{padding:20px 24px;max-width:1600px;margin:0 auto}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin-bottom:16px;overflow:hidden}}
.hd{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--bd);background:#141821}}
.doc{{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace}}
.badges .b{{display:inline-block;margin-left:6px;padding:2px 8px;border-radius:6px;background:#222834;color:var(--mut);font-size:12px}}
.badges .b.warn{{background:#3a1f22;color:var(--warn)}}
.cols{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}}
.col{{padding:12px 14px;border-right:1px solid var(--bd)}}
.col:last-child{{border-right:none}}
.lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin-bottom:6px}}
.txt{{white-space:normal;word-break:break-word}}
.txt.ai{{border-left:3px solid var(--ai);padding-left:10px}}
.txt.gen{{border-left:3px solid var(--gen);padding-left:10px}}
.txt.hu{{border-left:3px solid var(--hu);padding-left:10px}}
@media(max-width:1000px){{.cols{{grid-template-columns:1fr}}.col{{border-right:none;border-bottom:1px solid var(--bd)}}}}
</style></head><body>
<header><h1>Stage2 EOS 수정 리뷰 · {esc(args.tag)}</h1>
<div class="sum">{date} · ckpt <code>{esc(args.ckpt)}</code> · fusion={kind} · 디코딩 beam=4(max_length=1024, 반복방지 없음)<br>{summary}</div></header>
<div class="wrap">{''.join(cards)}</div>
</body></html>"""

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(page)
    print(f"리뷰 저장: {args.out}")
    print(f"요약: {summary}")


if __name__ == "__main__":
    main()
