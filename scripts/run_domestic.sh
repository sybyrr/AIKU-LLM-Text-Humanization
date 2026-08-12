#!/bin/bash
# 국내 모델 2종 추가 — kanana-1.5-8b(카카오) · Midm-2.0-Base-Instruct(KT)
#
# 둘 다 공식 GGUF 가 없어 fp16 을 받아 직접 변환한다.
#   HF fp16 → convert_hf_to_gguf.py (F16 GGUF) → llama-quantize (Q4_K_M)
#
# 목적: 현재 국내 모델이 LG(EXAONE) 하나뿐이라, "분량을 지키는 것이 EXAONE 특성인지
#       국내 모델 공통 특성인지" 가릴 수 없다. 카카오 8B 는 EXAONE-3.5-7.8B 와 같은
#       체급이라 직접 비교되고, KT 11.5B 는 Qwen3-14B 와 비슷한 체급이다.
set -u
W=/workspace
DS=$W/dataset
RV=$W/review
LG=$W/logs
cd "$W"
export LD_LIBRARY_PATH=/home/dev/bt/lib:/home/dev/bt/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
LC=/home/dev/llama.cpp
PY=/workspace/.venv/bin/python
M=/workspace/models
mkdir -p gen_full logs

wait_dl() {  # wait_dl <디렉터리> <최소GB> — hf download 완료 대기
  local n=0
  while true; do
    local gb=$(du -s --block-size=1G "$1" 2>/dev/null | cut -f1)
    [ -n "$gb" ] && [ "$gb" -ge "$2" ] && ! pgrep -f "hf download.*$(basename "$1")" >/dev/null && return 0
    n=$((n+1)); [ $n -gt 240 ] && { echo "  [$1] 다운로드 시간 초과"; return 1; }
    sleep 30
  done
}

prep() {  # prep <이름> <hf디렉터리> <최소GB>
  local name=$1 src=$2 gb=$3
  [ -s "$M/$name-Q4_K_M.gguf" ] && { echo "[$name] GGUF 이미 있음"; return 0; }
  echo "[$name] 다운로드 대기 $(date +%H:%M)"
  wait_dl "$src" "$gb" || return 1
  echo "[$name] F16 변환 $(date +%H:%M)"
  $PY $LC/convert_hf_to_gguf.py "$src" --outfile "$M/$name-F16.gguf" --outtype f16 \
    > "$LG/convert_$name.log" 2>&1 || { echo "[$name] 변환 실패"; tail -5 "$LG/convert_$name.log"; return 1; }
  echo "[$name] Q4_K_M 양자화 $(date +%H:%M)"
  $LC/build/bin/llama-quantize "$M/$name-F16.gguf" "$M/$name-Q4_K_M.gguf" Q4_K_M \
    > "$LG/quant_$name.log" 2>&1 || { echo "[$name] 양자화 실패"; tail -5 "$LG/quant_$name.log"; return 1; }
  rm -f "$M/$name-F16.gguf"          # F16 중간 파일은 크므로 삭제
  ls -lh "$M/$name-Q4_K_M.gguf" | awk '{print "  ["$9"] "$5}'
}

gen() {  # gen <포트> <GPU> <이름>
  local port=$1 gpus=$2 name=$3
  [ -s "$DS/generations/$name.jsonl" ] && { echo "[$name] 생성 이미 있음"; return; }
  [ -s "$M/$name-Q4_K_M.gguf" ] || { echo "[$name] GGUF 없음 — 건너뜀"; return; }
  CUDA_VISIBLE_DEVICES=$gpus $LC/build/bin/llama-server -m "$M/$name-Q4_K_M.gguf" \
    -ngl 99 -c 16384 -np 4 --host 127.0.0.1 --port "$port" > "$LG/server_$port.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do curl -s -m 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok && break; sleep 5; done
  curl -s -m 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok || {
    echo "[$name] 기동 실패"; tail -5 "$LG/server_$port.log"; kill $pid 2>/dev/null; return; }
  echo "[$name] 생성 시작 $(date +%H:%M)"
  $PY generate.py --in $DS/prompts/prompts_full.jsonl --out "$DS/generations/$name.jsonl" --model "$name" \
      --url "http://127.0.0.1:$port/v1/chat/completions" --slots 4 --temp 0.0 --no-think 2>&1 | sed "s/^/[$name] /"
  kill $pid 2>/dev/null; sleep 8
  echo "[$name] 완료 $(date +%H:%M)"
}

echo "=== 변환 $(date +%H:%M) ==="
prep kanana-8b $M/hf/kanana-8b 15 &
P1=$!
prep midm-11b  $M/hf/midm-11b  20 &
P2=$!
wait $P1 $P2

echo "=== 생성 $(date +%H:%M) ==="
gen 8080 0,1 kanana-8b &
gen 8081 2,3 midm-11b &
wait

echo "=== 완료 $(date +%H:%M) ==="
/home/dev/bin/duckdb -box -c "
SELECT model 모델, count(*) n, round(sum(gen_char)*1.0/sum(tokens),2) \"자/토큰\",
       round(avg(gen_char)) 평균자수, round(avg(gen_char*100.0/target_char)) 달성률,
       sum(has_markdown::INT) MD
FROM read_json('$DS/generations/*.jsonl', union_by_name=true) WHERE error IS NULL
GROUP BY model ORDER BY 달성률 DESC;"
