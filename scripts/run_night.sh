#!/bin/bash
# 야간 연쇄 실행 — run_full.sh 가 끝나면 이어서 돈다.
#
#   1) 거대모델 2종 추가 생성   Qwen3-30B-A3B(MoE) · gemma-3-27b-it
#   2) 양자화 비교              Qwen3-8B  Q4_K_M vs Q8_0  (조건 A 100편만)
#   3) Fast-DetectGPT 탐지      인간 100편 + 전체 생성물
#
# 각 단계는 산출물이 이미 있으면 건너뛰므로 중간에 죽어도 재실행 가능.
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
mkdir -p gen_full gen_quant logs

# run_full.sh 완료 대기
echo "[대기] 본 실행 완료 대기 $(date +%H:%M)"
while ! grep -q "^=== 완료" /tmp/full.log 2>/dev/null; do sleep 60; done
echo "[대기] 본 실행 완료 확인 $(date +%H:%M)"

serve() {  # serve <포트> <GPU> <gguf> — 준비될 때까지 대기, PID 반환
  CUDA_VISIBLE_DEVICES=$2 $LS -m "$3" -ngl 99 -c 16384 -np 4 \
    --host 127.0.0.1 --port "$1" > "$LG/server_$1.log" 2>&1 &
  echo $! > /tmp/pid_$1
  for _ in $(seq 1 240); do
    curl -s -m 3 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q ok && return 0
    sleep 5
  done
  return 1
}
unserve() { kill "$(cat /tmp/pid_$1 2>/dev/null)" 2>/dev/null; rm -f /tmp/pid_$1; sleep 8; }

ready() {  # ready <파일> <최소바이트>
  [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge "$2" ]
}

gen() {  # gen <포트> <GPU> <이름> <gguf> <최소바이트> <프롬프트> <출력>
  local port=$1 gpus=$2 name=$3 gguf=$4 minb=$5 pin=$6 pout=$7
  [ -s "$pout" ] && { echo "[$name] 이미 있음"; return; }
  local n=0
  while ! ready "$gguf" "$minb"; do n=$((n+1)); [ $n -gt 120 ] && { echo "[$name] 다운로드 미완 — 건너뜀"; return; }; sleep 30; done
  echo "[$name] 시작 $(date +%H:%M)"
  serve "$port" "$gpus" "$gguf" || { echo "[$name] 기동 실패"; return; }
  $PY generate.py --in "$pin" --out "$pout" --model "$name" \
      --url "http://127.0.0.1:$port/v1/chat/completions" \
      --slots 4 --temp 0.0 --no-think 2>&1 | sed "s/^/[$name] /"
  unserve "$port"
  echo "[$name] 완료 $(date +%H:%M)"
}

# ── 1. 거대모델 2종 (두 레인 병렬) ─────────────────────────────
echo "=== 1단계 거대모델 $(date +%H:%M) ==="
gen 8080 0,1 qwen3-30b-a3b $M/Qwen3-30B-A3B-Q4_K_M.gguf 17000000000 $DS/prompts/prompts_full.jsonl $DS/generations/qwen3-30b-a3b.jsonl &
gen 8081 2,3 gemma-3-27b   $M/gemma-3-27b-it-Q4_K_M.gguf 15000000000 $DS/prompts/prompts_full.jsonl $DS/generations/gemma-3-27b.jsonl &
wait

# ── 2. 양자화 비교 — Qwen3-8B Q4 vs Q8, 조건 A 100편 ───────────
echo "=== 2단계 양자화 비교 $(date +%H:%M) ==="
$DUCK -c "COPY (SELECT * FROM read_json('$DS/prompts/prompts_full.jsonl') WHERE cond='A') TO 'prompts_A100.jsonl' (FORMAT JSON);"
gen 8080 0,1 qwen3-8b-q8 $M/Qwen3-8B-Q8_0.gguf   8000000000 prompts_A100.jsonl $DS/generations/quant/qwen3-8b-q8.jsonl &
gen 8081 2,3 qwen3-8b-q4 $M/Qwen3-8B-Q4_K_M.gguf 4900000000 prompts_A100.jsonl $DS/generations/quant/qwen3-8b-q4.jsonl &
wait

# ── 3. Fast-DetectGPT ──────────────────────────────────────────
echo "=== 3단계 탐지기 $(date +%H:%M) ==="
cat $DS/generations/*.jsonl > $DS/generations/full_generations.jsonl
$PY -c "import transformers" 2>/dev/null || \
  (export PATH=\$HOME/.local/bin:\$PATH; uv pip install --python $PY transformers accelerate -q)
$PY detect_fastdetectgpt.py --scorer LGAI-EXAONE/EXAONE-4.0-1.2B \
    --human $DS/human/full_base.jsonl --gen $DS/generations/full_generations.jsonl \
    --out $DS/scores/detect_scores.jsonl --device cuda:0 2>&1 | tail -30

# ── 집계 ───────────────────────────────────────────────────────
echo "=== 전체 완료 $(date +%H:%M) ==="
$DUCK -box -c "
SELECT model 모델, cond 조건, count(*) n, round(avg(gen_char)) 평균자수,
       round(avg(gen_char*100.0/target_char)) 달성률, sum(has_markdown::INT) MD
FROM read_json('$DS/generations/*.jsonl') GROUP BY model, cond ORDER BY model, cond;"
echo "--- 양자화 비교 (조건 A) ---"
$DUCK -box -c "
SELECT model 모델, count(*) n, round(avg(gen_char)) 평균자수,
       round(avg(gen_char*100.0/target_char)) 달성률, round(avg(elapsed),1) 초
FROM read_json('$DS/generations/quant/*.jsonl') GROUP BY model ORDER BY model;" 2>/dev/null
