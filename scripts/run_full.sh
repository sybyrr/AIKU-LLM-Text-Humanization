#!/bin/bash
# 본 실행: 인간 원문 100편 × 조건 3개(A/E/C) × 모델 5종 = 1,500편
#
#   temperature 0.0  — 러셀 기본값과 일치(helpers.py model_call temp=0.0).
#                      파일럿에서 0.8 대비 첫문장 고유명사 33%→57%, 인용동사 13%→27%(인간 30%).
#   전부 Q4_K_M      — 양자화를 교란변수에서 제거. 다만 소형 모델에 상대적으로 불리하므로
#                      fp16 비교는 별도 과제로 남긴다.
#
# GPU 0·1 과 2·3 에 인스턴스를 하나씩 띄워 두 레인으로 병렬 실행한다(4·5는 사용 금지).
# 레인 배분은 소요 시간이 비슷하도록: 33B+14B  vs  32B+8B+7.8B
set -u
W=/workspace
DS=$W/dataset
RV=$W/review
LG=$W/logs
cd "$W"

export LD_LIBRARY_PATH=/home/dev/bt/lib:/home/dev/bt/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
LS=/home/dev/llama.cpp/build/bin/llama-server
PY=/workspace/.venv/bin/python
DUCK=/home/dev/bin/duckdb
M=/workspace/models
OUT=$DS/generations
mkdir -p $OUT logs

wait_file() {  # 다운로드가 아직 도는 경우 대비
  local n=0
  while [ ! -f "$1" ] || [ "$(stat -c%s "$1")" -lt "$2" ]; do
    n=$((n+1)); sleep 20
    [ $n -gt 180 ] && { echo "  [$(basename "$1")] 다운로드 시간 초과"; return 1; }
  done
  return 0
}

# lane <포트> <GPU> <이름> <gguf> <최소바이트>...  — 모델을 순서대로 처리
lane() {
  local port=$1 gpus=$2; shift 2
  local url="http://127.0.0.1:$port"
  while [ $# -gt 0 ]; do
    local name=$1 gguf=$2 minb=$3; shift 3
    [ -s "$OUT/$name.jsonl" ] && { echo "[$port] $name 이미 있음 — 건너뜀"; continue; }
    wait_file "$gguf" "$minb" || { echo "[$port] $name 파일 없음 — 건너뜀"; continue; }

    echo "[$port] $name 서버 기동"
    CUDA_VISIBLE_DEVICES=$gpus $LS -m "$gguf" -ngl 99 -c 16384 -np 4 \
      --host 127.0.0.1 --port "$port" > "$LG/server_$port.log" 2>&1 &
    local pid=$!
    local ok=0
    for _ in $(seq 1 240); do
      curl -s -m 3 "$url/health" 2>/dev/null | grep -q ok && { ok=1; break; }
      sleep 5
    done
    if [ $ok -eq 0 ]; then
      echo "[$port] $name 기동 실패"; kill $pid 2>/dev/null; continue
    fi

    echo "[$port] $name 생성 시작 $(date +%H:%M)"
    $PY generate.py --in $DS/prompts/prompts_full.jsonl --out "$OUT/$name.jsonl" \
        --model "$name" --url "$url/v1/chat/completions" \
        --slots 4 --temp 0.0 --no-think 2>&1 | sed "s/^/[$port] /"
    echo "[$port] $name 완료 $(date +%H:%M)"

    kill $pid 2>/dev/null; sleep 8
  done
}

# ── 준비 1. 조건 C 요약 100건 (Qwen3-14B) ──────────────────────
if [ ! -s $DS/prompts/full_summaries.jsonl ]; then
  echo "[준비] 요약 100건 생성 — Qwen3-14B"
  CUDA_VISIBLE_DEVICES=0,1 $LS -m $M/Qwen3-14B-Q4_K_M.gguf -ngl 99 -c 16384 -np 4 \
    --host 127.0.0.1 --port 8080 > $LG/server_sum.log 2>&1 &
  SPID=$!
  for _ in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; sleep 5; done
  $PY summarize.py --in $DS/human/full_base.jsonl --out $DS/prompts/full_summaries.jsonl --no-think 2>&1 | tail -3
  kill $SPID 2>/dev/null; sleep 8
else
  echo "[준비] 요약 이미 있음 — 건너뜀"
fi

# ── 준비 2. 프롬프트 300건 조립 ────────────────────────────────
echo "[준비] 프롬프트 조립"
$DUCK -c "
CREATE OR REPLACE MACRO head(d,p,t,kind,n) AS format(
     '당신은 {}자 {} {} 섹션의 기자입니다. 다음 {}에 해당하는 기사를 '
  || '약 {}자 분량으로 작성하세요. 일반 독자가 이해하기 쉽도록 간결하게 작성하세요. '
  || '제목은 다시 쓰지 말고 본문만 작성하고, 굵은 글씨·목록·소제목 같은 '
  || '마크다운 서식은 쓰지 마세요.', d,p,t,kind,n);
COPY (
  SELECT doc_id,'A' cond,target_char,n_char,title,
         head(date_ko,publisher,topic,'제목',target_char)
         ||format(chr(10)||chr(10)||'제목: {}',title) prompt
  FROM read_json('$DS/human/full_base.jsonl')
  UNION ALL
  SELECT b.doc_id,'C',b.target_char,b.n_char,b.title,
         head(b.date_ko,b.publisher,b.topic,'제목과 요약',b.target_char)
         ||format(chr(10)||chr(10)||'제목: {}'||chr(10)||'요약: {}',b.title,s.summary)
  FROM read_json('$DS/human/full_base.jsonl') b JOIN read_json('$DS/prompts/full_summaries.jsonl') s USING (doc_id)
  UNION ALL
  SELECT b.doc_id,'E',b.target_char,b.n_char,b.title,
         head(b.date_ko,b.publisher,b.topic,'제목과 핵심어',b.target_char)
         ||format(chr(10)||chr(10)||'제목: {}'||chr(10)||'핵심어: {}',b.title,e.nouns)
  FROM read_json('$DS/human/full_base.jsonl') b JOIN read_json('$DS/prompts/full_nouns.jsonl') e USING (doc_id)
  ORDER BY cond, doc_id
) TO '$DS/prompts/prompts_full.jsonl' (FORMAT JSON);"
echo "[준비] 프롬프트 $(wc -l < $DS/prompts/prompts_full.jsonl)건"

# ── 생성: 두 레인 병렬 ─────────────────────────────────────────
echo "=== 생성 시작 $(date +%H:%M) ==="
lane 8080 0,1 \
  exaone-33b $M/EXAONE-4.5-33B-Q4_K_M.gguf   19000000000 \
  qwen3-14b  $M/Qwen3-14B-Q4_K_M.gguf         8300000000 &
L1=$!
lane 8081 2,3 \
  qwen3-32b   $M/Qwen3-32B-Q4_K_M.gguf       19000000000 \
  qwen3-8b    $M/Qwen3-8B-Q4_K_M.gguf         4900000000 \
  exaone-7.8b $M/EXAONE-3.5-7.8B-Q4_K_M.gguf  4400000000 &
L2=$!
wait $L1 $L2

cat $OUT/*.jsonl > $DS/generations/full_generations.jsonl
echo "=== 완료 $(date +%H:%M) — $(wc -l < $DS/generations/full_generations.jsonl)편 ==="
$DUCK -box -c "
SELECT model 모델, cond 조건, count(*) n, round(avg(gen_char)) 평균자수,
       round(avg(gen_char*100.0/target_char)) 달성률, sum(has_markdown::INT) MD
FROM read_json('$OUT/*.jsonl') GROUP BY model, cond ORDER BY model, cond;"
