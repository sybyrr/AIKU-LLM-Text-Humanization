#!/usr/bin/env bash
# MASH 본 구축 — 인간 풀 전량에 P3b 재서술 생성.
#
# 수 시간짜리 작업이라 --resume 전제로 돌린다. 중간에 죽거나 서버가 내려가면
# 같은 명령을 다시 실행하면 된다(성공분은 건너뛴다).
#
# 사용: nohup bash scripts/run_mash_full.sh > logs/gen_full_p3b.log 2>&1 &
set -uo pipefail
cd /workspace

PY=/workspace/.venv/bin/python3
IN=mash/prompts_full_p3b.jsonl
OUT=mash/gen_full_p3b.jsonl
URL=http://127.0.0.1:8081/v1/chat/completions

# 서버가 내려가 있으면 재시도해도 소용없다 — 먼저 확인한다.
if ! curl -s -m 5 "${URL%/v1/*}/health" | grep -q ok; then
  echo "llama-server(8081)가 응답하지 않는다. 먼저 기동할 것:"
  echo "  docker exec -d gpu1 bash /workspace/scripts/gpu_serve.sh \\"
  echo "    /workspace/models/Qwen3-8B-Q4_K_M.gguf 8081 16384"
  exit 1
fi

# 일시적 서버 오류로 죽어도 스스로 몇 번 이어붙인다.
for attempt in 1 2 3 4 5; do
  echo "===== 시도 $attempt · $(date '+%F %T') ====="
  "$PY" scripts/generate.py \
    --in "$IN" --out "$OUT" --url "$URL" \
    --model qwen3-8b --no-think --temp 0.8 --seed 42 --resume && break
  echo "중단됨 — 30초 후 이어서 재시도"
  sleep 30
done

echo "===== 종료 $(date '+%F %T') · 산출 $(wc -l < "$OUT") 건 ====="
