#!/bin/bash
# 파일럿: 인간 원문 10편 × 조건 3개 × 모델 3종 = 90편 생성
#
#   조건 A  제목만              (프롬프트 34자,  기사 대비 1.9%)
#   조건 B  제목 + 도입부 2문장 (152자, 8.5%)  ← 인간 산문이 들어가는 유일한 조건
#   조건 C  제목 + 요약 1문장   (55자, 3.1%)   ← 러셀 부제 비율 2.9% 대응
#
# 조건 C 의 요약은 Qwen3-14B 로 한 번만 만들어 모든 모델이 공유한다(조건을 상수로).
# 요약기를 EXAONE 이 아니라 Qwen 으로 둔 이유: 생성 모델 3종 중 EXAONE 계열이 2개라
# EXAONE 으로 요약하면 3분의 2가 자기 계열 요약을 받게 된다.
set -e
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/dev/bt/lib:/home/dev/bt/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
LS=/home/dev/llama.cpp/build/bin/llama-server
PY=/workspace/.venv/bin/python
DUCK=/home/dev/bin/duckdb
M=/workspace/models
OUT=gen_aec
mkdir -p $OUT

# GPU 0,1 만 사용 (4,5 는 허가 필요, 2,3 은 예비)
GPUS=${GPUS:-0,1}
PORT=${PORT:-8080}
URL=http://127.0.0.1:$PORT

start() {  # start <gguf> <extra args...>
  local gguf=$1; shift
  CUDA_VISIBLE_DEVICES=$GPUS $LS -m "$gguf" -ngl 99 -c 16384 -np 4 \
    --host 127.0.0.1 --port $PORT "$@" > /tmp/server_$PORT.log 2>&1 &
  echo $! > /tmp/server_$PORT.pid
  echo -n "  서버 기동"
  for _ in $(seq 1 180); do
    curl -s $URL/health 2>/dev/null | grep -q ok && { echo " 준비됨"; return; }
    sleep 5; echo -n "."
  done
  echo " 실패"; tail -20 /tmp/server_$PORT.log; exit 1
}
stop() {
  [ -f /tmp/server_$PORT.pid ] && kill "$(cat /tmp/server_$PORT.pid)" 2>/dev/null || true
  rm -f /tmp/server_$PORT.pid
  sleep 5
}
trap stop EXIT

# ── 3. 모델별 생성 ─────────────────────────────────────────────
echo "생성 (조건 A/E/C × 모델 3종 = 90편)"
wait_file() {  # wait_file <경로> <최소바이트> — 다운로드가 아직 도는 경우 대비
  local n=0
  while [ ! -f "$1" ] || [ "$(stat -c%s "$1")" -lt "$2" ]; do
    [ $n -eq 0 ] && echo -n "  $(basename "$1") 다운로드 대기"
    echo -n "."; n=$((n+1)); sleep 20
    [ $n -gt 180 ] && { echo " 시간 초과"; exit 1; }
  done
  [ $n -gt 0 ] && echo " 완료"
  return 0
}

run_model() {  # run_model <이름> <gguf> <최소바이트> [--no-think]
  echo "── $1"
  wait_file "$2" "$3"
  start "$2"
  $PY generate.py --in prompts_AEC.jsonl --out $OUT/$1.jsonl \
       --model "$1" --url $URL/v1/chat/completions --slots 4 ${4:-}
  stop
}
run_model exaone-33b  $M/EXAONE-4.5-33B-Q4_K_M.gguf   19000000000 --no-think
run_model qwen3-14b   $M/Qwen3-14B-Q4_K_M.gguf        8900000000 --no-think
run_model exaone-7.8b $M/EXAONE-3.5-7.8B-Q4_K_M.gguf  4700000000

cat $OUT/*.jsonl > pilot_generations.jsonl
echo
echo "완료: pilot_generations.jsonl ($(wc -l < pilot_generations.jsonl)건)"
