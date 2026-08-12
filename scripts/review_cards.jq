def d: tostring | .[0:4] + "-" + .[4:6] + "-" + .[6:8];
def commafy: tostring | explode | reverse | to_entries
  | map(if (.key > 0 and .key % 3 == 0) then [44, .value] else [.value] end)
  | flatten | reverse | implode;
# 본문은 문단 단위로 저장돼 있다 (개행 2개 구분). 문단마다 <p> 로 감싼다.
def paras: split("\n\n") | map("<p>" + (. | @html) + "</p>") | join("\n      ");

.[] |
"<article class=\"card\">
  <div class=\"meta\">
    <span class=\"pub\">\(.publisher | @html)</span>
    <span class=\"topic\">\(.topic | @html)</span>
    <span>\(.date | d)</span>
    <span>\(.n_char | commafy)자 · \(.n_para)문단</span>
    <code>\(.doc_id | @html)</code>
  </div>
  <h2>\(.title | @html)</h2>
  <div class=\"prompt\">
    <div class=\"prompt-label\">러셀 프롬프트 — 목표 \(.target_char | commafy)자</div>
    <pre>\(.prompt | @html)</pre>
  </div>
  <div class=\"body\">
      \(.body | paras)
  </div>
</article>"
