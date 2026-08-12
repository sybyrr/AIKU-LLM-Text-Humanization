#!/bin/sh
# 추출 표본 육안 검수용 HTML 생성
#   사용법: sh make_review_html.sh <sample.json> <out.html>
#   입력은 extract_russell_pool.sql 산출물을 조인한 JSON 배열
set -e
IN="${1:-sample10.json}"
OUT="${2:-review.html}"
DIR=$(dirname "$0")
N=$(jq 'length' "$IN")

{
cat <<'HEAD'
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>러셀 표본 검수 — 신문 말뭉치 2023</title>
<style>
:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a1a; --muted: #6b6b6b;
  --line: #e3e0da; --accent: #7a5c3e; --chip: #f0ece5; --code: #f6f4f0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181c; --panel: #1d2025; --ink: #e8e6e3; --muted: #9a9691;
    --line: #2f333a; --accent: #c9a882; --chip: #262a30; --code: #14161a;
  }
}
:root[data-theme="dark"] {
  --bg: #16181c; --panel: #1d2025; --ink: #e8e6e3; --muted: #9a9691;
  --line: #2f333a; --accent: #c9a882; --chip: #262a30; --code: #14161a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
    "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif;
  line-height: 1.75; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 820px; margin: 0 auto; padding: 3rem 1.25rem 5rem; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 1.25rem; margin-bottom: 2rem; }
h1 { font-size: 1.6rem; margin: 0 0 .4rem; letter-spacing: -.02em; }
.sub { color: var(--muted); font-size: .9rem; margin: 0; }
.facts { margin-top: 1rem; font-size: .82rem; color: var(--muted); }
.facts div { padding: .15rem 0; }
.facts b { color: var(--ink); font-weight: 600; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.5rem 1.6rem; margin-bottom: 1.5rem;
}
.meta {
  display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
  font-size: .76rem; color: var(--muted); margin-bottom: .7rem;
}
.meta .pub { background: var(--accent); color: var(--bg); font-weight: 700;
  padding: .12rem .5rem; border-radius: 4px; font-size: .74rem; }
.meta .topic { background: var(--chip); padding: .12rem .5rem; border-radius: 4px; }
.meta code { font-size: .7rem; opacity: .65; margin-left: auto; }
h2 { font-size: 1.22rem; line-height: 1.45; margin: 0 0 1rem; letter-spacing: -.015em; }
.prompt { background: var(--code); border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0; padding: .8rem 1rem; margin-bottom: 1.1rem; }
.prompt-label { font-size: .72rem; font-weight: 700; color: var(--accent);
  letter-spacing: .04em; margin-bottom: .4rem; }
.prompt pre { margin: 0; font-size: .82rem; line-height: 1.65; color: var(--muted);
  white-space: pre-wrap; word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.body { font-size: .97rem; line-height: 1.95; text-align: justify; word-break: keep-all; }
.body p { margin: 0 0 1.05em; }
.body p:last-child { margin-bottom: 0; }
footer { margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
  font-size: .8rem; color: var(--muted); }
footer p { margin: .35rem 0; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>러셀 방식 인간 원문 — 표본 검수</h1>
  <p class="sub">국립국어원 신문 말뭉치 2023 (2022년 생산 기사)</p>
  <div class="facts">
HEAD

cat <<FACTS
    <div><b>표본</b> 전체 100편 중 $N편 (토픽별 2편)</div>
    <div><b>필터</b> 발행일 &lt; 2022-11-01 · 토픽 5종 · 본문 1,500~2,500자 · 8문단 이상</div>
    <div><b>조건 통과 후보</b> 20,739편 (샤드 1·7·8 기준)</div>
    <div><b>파일럿 출처</b> HF 미러 Saxo/ko-news-corpus-{1,7,8} — 재배포 조건 확인 필요</div>
FACTS

echo '  </div></header><main>'
jq -r -f "$DIR/review_cards.jq" "$IN"
cat <<'FOOT'
</main>
<footer>
  <p><b>검수 포인트</b> — ① 첫 문장이 제목으로 제대로 잡혔는지 ② 본문에 사진 캡션·기자 이메일·제보 안내가 섞이지 않았는지 ③ 구두점이 원형대로인지 ④ 단신이 아니라 서술형 기사인지</p>
  <p><b>문단 구조</b> — 컬럼명은 sentence 지만 실제 단위는 문단이다(본문 행의 29.1%가 문장 2개 이상). 원문 문단 경계가 살아 있어 그대로 복원했다.</p>
</footer>
</div>
</body>
</html>
FOOT
} > "$OUT"

echo "wrote $OUT ($N articles, $(wc -c < "$OUT") bytes)"
