#!/bin/sh
# 파일럿 결과 검수 HTML — 기사별로 인간 원문과 (모델 × 조건) 생성물을 나란히 놓는다.
#   사용: sh make_pilot_html.sh [out.html]
set -e
cd "$(dirname "$0")"
OUT="${1:-pilot_review.html}"
DUCK=/home/dev/bin/duckdb

$DUCK -c "
COPY (
  WITH g AS (SELECT * FROM read_json('pilot_generations.jsonl')),
       b AS (SELECT * FROM read_json('pilot_base.jsonl')),
       s AS (SELECT * FROM read_json('pilot_summaries.jsonl'))
  SELECT b.doc_id, b.title, b.publisher, b.topic, b.date_ko, b.n_char AS human_char,
         b.target_char, b.body, s.summary,
         list(struct_pack(model := g.model, cond := g.cond, gen_char := g.gen_char,
                          ratio := round(g.gen_char*100.0/g.target_char),
                          md := g.has_markdown, think := g.had_think,
                          secs := g.elapsed, text := g.text)
              ORDER BY g.cond, g.model) AS gens
  FROM b JOIN g USING (doc_id) JOIN s USING (doc_id)
  GROUP BY ALL ORDER BY b.doc_id
) TO 'pilot_cards.json' (FORMAT JSON, ARRAY true);"

{
cat <<'HEAD'
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>파일럿 검수 — 모델 3종 × 조건 3개</title>
<style>
:root { --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e3e0da;
        --accent:#7a5c3e; --chip:#f0ece5; --code:#f6f4f0; --human:#2f6f4f; --warn:#a8442a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#16181c; --panel:#1d2025; --ink:#e8e6e3; --muted:#9a9691; --line:#2f333a;
  --accent:#c9a882; --chip:#262a30; --code:#14161a; --human:#7fbf9a; --warn:#e08a6a; } }
:root[data-theme="dark"] { --bg:#16181c; --panel:#1d2025; --ink:#e8e6e3; --muted:#9a9691;
  --line:#2f333a; --accent:#c9a882; --chip:#262a30; --code:#14161a; --human:#7fbf9a; --warn:#e08a6a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.75;
  font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Noto Sans KR",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:2.5rem 1.25rem 5rem; }
header { border-bottom:2px solid var(--ink); padding-bottom:1rem; margin-bottom:1.5rem; }
h1 { font-size:1.5rem; margin:0 0 .3rem; letter-spacing:-.02em; }
.sub { color:var(--muted); font-size:.85rem; margin:0; }
.legend { margin-top:.9rem; font-size:.8rem; color:var(--muted); }
.legend b { color:var(--ink); }
.art { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:1.4rem 1.5rem; margin-bottom:1.6rem; }
.meta { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; font-size:.75rem;
  color:var(--muted); margin-bottom:.6rem; }
.meta .pub { background:var(--accent); color:var(--bg); font-weight:700; padding:.1rem .5rem; border-radius:4px; }
.meta .topic { background:var(--chip); padding:.1rem .5rem; border-radius:4px; }
.meta code { margin-left:auto; opacity:.6; font-size:.7rem; }
h2 { font-size:1.15rem; line-height:1.45; margin:0 0 .8rem; letter-spacing:-.015em; }
.sum { background:var(--code); border-left:3px solid var(--accent); border-radius:0 6px 6px 0;
  padding:.6rem .9rem; margin-bottom:1rem; font-size:.85rem; color:var(--muted); }
.sum b { color:var(--accent); font-size:.72rem; letter-spacing:.04em; }
details { border-top:1px solid var(--line); }
summary { cursor:pointer; padding:.6rem 0; font-size:.85rem; font-weight:600; list-style:none; }
summary::-webkit-details-marker { display:none; }
summary::before { content:"▸ "; color:var(--muted); }
details[open] summary::before { content:"▾ "; }
.tag { display:inline-block; font-size:.7rem; font-weight:600; padding:.08rem .45rem;
  border-radius:4px; background:var(--chip); color:var(--muted); margin-left:.35rem; }
.tag.bad { background:var(--warn); color:var(--bg); }
.tag.hum { background:var(--human); color:var(--bg); }
.txt { font-size:.92rem; line-height:1.9; text-align:justify; word-break:keep-all;
  padding:.3rem 0 1rem; }
.txt p { margin:0 0 .95em; }
</style>
</head>
<body><div class="wrap">
<header>
  <h1>파일럿 검수 — 인간 원문 vs 생성물</h1>
  <p class="sub">10편 × 조건 3개(A 제목만 / B 제목+도입부 / C 제목+요약) × 모델 3종</p>
  <div class="legend">
    <b>검수 포인트</b> — 기사체(~다) 유지 · 조사 오류 · 번역체 구문 · 분량 달성률 ·
    마크다운 누출 · 지어낸 인명·기관이 한국어로 그럴듯한지<br>
    <b>태그</b> — 달성률은 목표 글자수 대비, <span class="tag bad">MD</span> 는 마크다운 검출,
    <span class="tag bad">think</span> 는 사고블록 누출
  </div>
</header>
HEAD

jq -r '
def esc: @html;
def paras: split("\n") | map(select(length>0)) | map("<p>"+(.|esc)+"</p>") | join("");
.[] |
"<article class=\"art\">
  <div class=\"meta\"><span class=\"pub\">\(.publisher|esc)</span>
    <span class=\"topic\">\(.topic|esc)</span><span>\(.date_ko|esc)</span>
    <span>원문 \(.human_char)자 · 목표 \(.target_char)자</span>
    <code>\(.doc_id|esc)</code></div>
  <h2>\(.title|esc)</h2>
  <div class=\"sum\"><b>조건 C 요약</b><br>\(.summary|esc)</div>
  <details open><summary>인간 원문<span class=\"tag hum\">HUMAN</span>
    <span class=\"tag\">\(.human_char)자</span></summary>
    <div class=\"txt\">\(.body|paras)</div></details>" +
( [ .gens[] |
  "<details><summary>\(.cond) · \(.model|esc)<span class=\"tag\">\(.gen_char)자 · \(.ratio)%</span>" +
  (if .md then "<span class=\"tag bad\">MD</span>" else "" end) +
  (if .think then "<span class=\"tag bad\">think</span>" else "" end) +
  "<span class=\"tag\">\(.secs)초</span></summary>
   <div class=\"txt\">\(.text|paras)</div></details>" ] | join("") ) +
"</article>"' pilot_cards.json

echo '</div></body></html>'
} > "$OUT"

echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
