#!/usr/bin/env bash
# 준비된 shard를 GPU별로 실행합니다. 각 shard는 Binoculars 후 FastDetectGPT를 채점합니다.
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "사용법: $0 WORK_DIR GPU_ID [GPU_ID ...]" >&2
  exit 2
fi

WORK_DIR=$1
shift
GPU_IDS=("$@")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
SCORER=$SCRIPT_DIR/zero_shot_matrix.py
WORKERS=$WORK_DIR/workers
STATUS=$WORK_DIR/STATUS
mkdir -p "$WORKERS"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

SHARDS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["shards"])' \
    "$WORK_DIR/inputs/manifest.json"
) || exit 1
if [ "${#GPU_IDS[@]}" -ne "$SHARDS" ]; then
  echo "manifest shard 수($SHARDS)와 GPU 수(${#GPU_IDS[@]})가 다릅니다." >&2
  exit 2
fi

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  gpu=${GPU_IDS[$shard]}
  (
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" score \
      --detector binoculars --shard "$shard" --device cuda:0 || exit 1
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" score \
      --detector fastdetectgpt --shard "$shard" --device cuda:0
  ) >> "$WORKERS/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "$!" > "$WORKERS/shard_${shard}.pid"
done

echo "RUNNING $(stamp) workers=${pids[*]}" > "$STATUS"
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "FAILED $(stamp) stage=scoring" > "$STATUS"
  exit 1
fi
"$PYTHON" "$SCORER" --work-dir "$WORK_DIR" status > "$WORK_DIR/final_status.json"
echo "SCORED $(stamp)" > "$STATUS"
