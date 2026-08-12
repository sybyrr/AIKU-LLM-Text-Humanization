#!/usr/bin/env python3
"""보존 게이트 검토용 정적 HTML 생성.

목적: 자동 보존 게이트가 과한지 사람이 판단하게 한다.
현재 게이트는 수치·방향어·고유명사·길이를 동시에 요구해서 통과율이 낮은데,
그중 실제로 "내용이 훼손됐다"고 볼 만한 것이 몇 건인지는 읽어봐야 안다.
이 판단 하나로 최종 수율이 크게 움직인다.

사용:
  python3 stage0_make_review.py --human pilot/data/seed_abstract.jsonl \
      --gen pilot/gen/abstract_qwen3-8b.jsonl --qa pilot/qa/abstract_qwen3-8b.jsonl \
      --out review/pair_review_qwen.html
"""
import argparse
import html
import json
import re
from pathlib import Path

NUM = re.compile(r"\d+(?:[.,]\d+)*\s*(?:%|퍼센트|명|건|개|년|월|일|시간|배|원|억|만|천)?")


def mark_numbers(text, missing_set):
    """수치를 강조. 원문에만 있고 재서술문에 없는 것은 빨갛게."""
    out, last = [], 0
    for m in NUM.finditer(text):
        out.append(html.escape(text[last : m.start()]))
        tok = m.group(0)
        key = re.sub(r"\s+", "", tok)
        cls = "num miss" if key in missing_set else "num"
        out.append(f'<span class="{cls}">{html.escape(tok)}</span>')
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e0e0e0;--ok:#0a7d3f;--bad:#c62828;--warn:#b26a00}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c2f36}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
 font:15px/1.7 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mut);margin-bottom:20px;font-size:14px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);
 margin-bottom:16px;z-index:5;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.bar label{font-size:14px;cursor:pointer}
.card{border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
.head{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline;margin-bottom:10px;font-size:13px}
.id{font-weight:600}
.tag{padding:1px 7px;border-radius:10px;border:1px solid var(--line);color:var(--mut)}
.pass{color:var(--ok);border-color:var(--ok)}
.fail{color:var(--bad);border-color:var(--bad)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.col h3{font-size:12px;color:var(--mut);margin:0 0 6px;text-transform:uppercase;letter-spacing:.05em}
.txt{border:1px solid var(--line);border-radius:6px;padding:10px;max-height:340px;overflow:auto}
.num{background:rgba(255,214,0,.28);border-radius:3px;padding:0 2px}
.num.miss{background:rgba(198,40,40,.28);text-decoration:line-through}
.why{margin-top:8px;font-size:13px;color:var(--warn)}
.metrics{font-size:12px;color:var(--mut);margin-top:6px}
.vote{margin-top:10px;font-size:13px;color:var(--mut)}
.vote b{color:var(--fg)}
"""

JS = """
function apply(){
  const only=document.getElementById('onlyfail').checked;
  const v=document.getElementById('variant').value;
  document.querySelectorAll('.card').forEach(c=>{
    const okFail = !only || c.dataset.pass==='0';
    const okVar = (v==='all') || (c.dataset.variant===v);
    c.style.display = (okFail&&okVar)?'':'none';
  });
  const n=[...document.querySelectorAll('.card')].filter(c=>c.style.display!=='none').length;
  document.getElementById('count').textContent=n+'건 표시';
}
document.addEventListener('DOMContentLoaded',apply);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--gen", action="append", required=True)
    ap.add_argument("--qa", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="재서술 짝 검토")
    a = ap.parse_args()

    humans = {json.loads(l)["pair_id"]: json.loads(l) for l in open(a.human, encoding="utf-8") if l.strip()}
    gens, qas = {}, {}
    for p in a.gen:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                g = json.loads(l)
                if g.get("text"):
                    gens[(g["doc_id"], g["cond"], g["model"])] = g
    for p in a.qa:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                q = json.loads(l)
                qas[(q["pair_id"], q["variant"], q["model"])] = q

    variants = sorted({k[1] for k in qas})
    cards = []
    for key in sorted(qas, key=lambda k: (k[2], k[1], k[0])):
        q, g = qas[key], gens.get(key)
        if not g:
            continue
        h = humans[key[0]]
        hn = {re.sub(r"\s+", "", t) for t in NUM.findall(h["human_text"])}
        an = {re.sub(r"\s+", "", t) for t in NUM.findall(g["text"])}
        missing = hn - an

        why = []
        if q["nums_missing"]:
            why.append(f"수치 {q['nums_missing']}개 누락")
        if q["nums_hallucinated"]:
            why.append(f"없던 수치 {q['nums_hallucinated']}개 생김")
        if q["direction_delta"]:
            why.append("방향어 변화: " + ", ".join(f"{k}{v:+d}" for k, v in q["direction_delta"].items()))
        if not 80 <= q["len_ratio"] <= 130:
            why.append(f"길이 {q['len_ratio']}%")
        if q["propn_keep_pct"] is not None and q["propn_keep_pct"] < 80:
            why.append(f"고유명사 보존 {q['propn_keep_pct']}%")
        if q["has_markdown"]:
            why.append("마크다운 유출")

        cards.append(f"""
<div class="card" data-pass="{1 if q['preserved'] else 0}" data-variant="{html.escape(q['variant'])}">
  <div class="head">
    <span class="id">{html.escape(q['pair_id'])}</span>
    <span class="tag">{html.escape(q['model'])}</span>
    <span class="tag">{html.escape(q['variant'])}</span>
    <span class="tag {'pass' if q['preserved'] else 'fail'}">{'게이트 통과' if q['preserved'] else '게이트 탈락'}</span>
  </div>
  <div class="cols">
    <div class="col"><h3>인간 원문 ({q['human_chars']}자)</h3>
      <div class="txt">{mark_numbers(h['human_text'], missing)}</div></div>
    <div class="col"><h3>AI 재서술 ({q['ai_chars']}자)</h3>
      <div class="txt">{mark_numbers(g['text'], set())}</div></div>
  </div>
  {'<div class="why">탈락 사유 — ' + html.escape(" · ".join(why)) + "</div>" if why else ""}
  <div class="metrics">길이 {q['len_ratio']}% · 최장 공통 {q['lcs']}자 · 12자 겹침 {q['ngram_overlap']}%
    · 고유명사 보존 {q['propn_keep_pct'] if q['propn_keep_pct'] is not None else '-'}%</div>
  <div class="vote">판단: 이 재서술은 원문의 <b>내용을 보존</b>했는가? (게이트가 과하다면 통과로 봐야 함)</div>
</div>""")

    opts = "".join(f'<option value="{v}">{v}</option>' for v in variants)
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.title)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(a.title)}</h1>
<div class="sub">노란색 = 수치 · <span style="background:rgba(198,40,40,.28)">빨간 취소선</span> = 재서술문에서 사라진 수치.
자동 게이트가 과한지 판단하는 것이 목적입니다.</div>
<div class="bar">
  <label><input type="checkbox" id="onlyfail" onchange="apply()"> 탈락 건만 보기</label>
  <label>변형 <select id="variant" onchange="apply()"><option value="all">전체</option>{opts}</select></label>
  <span id="count"></span>
</div>
{''.join(cards)}
<script>{JS}</script></body></html>"""

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"{len(cards)}건 → {out}")


if __name__ == "__main__":
    main()
