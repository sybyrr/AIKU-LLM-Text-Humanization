#!/bin/sh
# 본 실행 검수 HTML — 기사 1편당 인간 원문 + 모델 7종 × 조건 3개 생성물을 나란히.
#   사용: sh make_full_html.sh [in.json] [out.html]
set -e
W=/workspace
DS=$W/dataset
RV=$W/review
LG=$W/logs
cd "$W"
IN="${1:-$RV/full_cards.json}"; OUT="${2:-$RV/full_review.html}"

{
cat <<'HEAD'
<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>본 실행 검수 — 모델 7종 × 조건 3</title>
<style>
:root{--bg:#f7f8f9;--panel:#fff;--ink:#191d24;--muted:#6d7480;--line:#e0e4ea;
 --accent:#27527f;--chip:#eef1f5;--code:#f3f5f8;--human:#2f6248;--warn:#8c4636}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --bg:#15181d;--panel:#1c2026;--ink:#e5e8ee;--muted:#8a919c;--line:#2c323a;
 --accent:#7fb0e0;--chip:#242931;--code:#171b20;--human:#7cc3a0;--warn:#d9927f}}
:root[data-theme="dark"]{--bg:#15181d;--panel:#1c2026;--ink:#e5e8ee;--muted:#8a919c;
 --line:#2c323a;--accent:#7fb0e0;--chip:#242931;--code:#171b20;--human:#7cc3a0;--warn:#d9927f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.75;
 font-family:"Apple SD Gothic Neo","Pretendard","Noto Sans KR",-apple-system,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:2.5rem 1.25rem 5rem}
header{border-bottom:2px solid var(--ink);padding-bottom:1rem;margin-bottom:1.6rem}
h1{font-size:1.45rem;margin:0 0 .35rem;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:.85rem;margin:0}
.legend{margin-top:.8rem;font-size:.78rem;color:var(--muted)}
.art{background:var(--panel);border:1px solid var(--line);border-radius:9px;
 padding:1.3rem 1.5rem;margin-bottom:1.5rem}
.meta{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;font-size:.74rem;
 color:var(--muted);margin-bottom:.55rem}
.meta .pub{background:var(--accent);color:var(--bg);font-weight:700;padding:.1rem .5rem;border-radius:4px}
.meta .topic{background:var(--chip);padding:.1rem .5rem;border-radius:4px}
.meta code{margin-left:auto;opacity:.6;font-size:.7rem}
h2{font-size:1.12rem;line-height:1.45;margin:0 0 .85rem;letter-spacing:-.015em}
.ctx{background:var(--code);border-left:3px solid var(--accent);border-radius:0 5px 5px 0;
 padding:.55rem .9rem;margin-bottom:.5rem;font-size:.82rem;color:var(--muted)}
.ctx b{color:var(--accent);font-size:.7rem;letter-spacing:.05em;display:block;margin-bottom:.2rem}
details{border-top:1px solid var(--line)}
summary{cursor:pointer;padding:.55rem 0;font-size:.84rem;font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--muted)}
details[open] summary::before{content:"▾ "}
.tag{display:inline-block;font-size:.69rem;font-weight:600;padding:.06rem .42rem;
 border-radius:4px;background:var(--chip);color:var(--muted);margin-left:.3rem;
 font-variant-numeric:tabular-nums}
.tag.bad{background:var(--warn);color:var(--bg)}
.tag.hum{background:var(--human);color:var(--bg)}
.txt{font-size:.9rem;line-height:1.9;text-align:justify;word-break:keep-all;padding:.25rem 0 1rem}
.txt p{margin:0 0 .9em}
</style></head><body><div class="wrap">
<header><h1>본 실행 검수 — 인간 원문 vs 생성물 21종</h1>
<p class="sub">모델 7종 × 조건 3개(A 제목만 / E 명사 25개 / C 요약 1문장) · temperature 0.0 · Q4_K_M</p>
<div class="legend">태그는 <b>글자수 · 목표 대비 달성률 · 탐지 점수</b> 순. 탐지 임계값 0.126(인간 FPR 5%)을 넘으면 AI로 판정됩니다.
<span class="tag bad">MD</span>는 마크다운 검출.</div></header>
HEAD

jq -r '
def esc: @html;
def paras: split("\n")|map(select(length>0))|map("<p>"+(.|esc)+"</p>")|join("");
.[] |
"<article class=\"art\">
<div class=\"meta\"><span class=\"pub\">\(.publisher|esc)</span>
<span class=\"topic\">\(.topic|esc)</span><span>\(.date_ko|esc)</span>
<span>원문 \(.human_char)자 · 목표 \(.target_char)자</span><code>\(.doc_id|esc)</code></div>
<h2>\(.title|esc)</h2>
<div class=\"ctx\"><b>조건 C 요약</b>\(.summary|esc)</div>
<div class=\"ctx\"><b>조건 E 핵심어</b>\(.nouns|esc)</div>
<details open><summary>인간 원문<span class=\"tag hum\">HUMAN</span><span class=\"tag\">\(.human_char)자</span><span class=\"tag\">점수 \(.human_score*1000|round/1000)</span></summary>
<div class=\"txt\">\(.body|paras)</div></details>" +
([.gens[]|
 "<details><summary>\(.cond) · \(.model|esc)<span class=\"tag\">\(.gen_char)자 · \(.ratio)%</span><span class=\"tag\">점수 \(.score)</span>"
 + (if .md then "<span class=\"tag bad\">MD</span>" else "" end)
 + "</summary><div class=\"txt\">\(.text|paras)</div></details>"]|join(""))
+ "</article>"' "$IN"

echo '</div></body></html>'
} > "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
