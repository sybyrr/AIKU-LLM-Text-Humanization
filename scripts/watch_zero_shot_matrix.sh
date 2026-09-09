#!/usr/bin/env bash
# shard worker 종료를 감지하고 append된 마지막 행부터 자동 재개합니다.
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
POLL_SECONDS=${POLL_SECONDS:-60}
SCORER=$SCRIPT_DIR/zero_shot_matrix.py
SCORES=$WORK_DIR/scores
WORKERS=$WORK_DIR/workers
STATUS=$WORK_DIR/WATCH_STATUS
LOG=$WORK_DIR/watchdog.log
mkdir -p "$SCORES" "$WORKERS"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

SHARDS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["shards"])' \
    "$WORK_DIR/inputs/manifest.json"
) || exit 1
if [ "${#GPU_IDS[@]}" -ne "$SHARDS" ]; then
  echo "manifest shard 수($SHARDS)와 GPU 수(${#GPU_IDS[@]})가 다릅니다." >&2
  exit 2
fi

detector_done() {
  [ -f "$SCORES/${1}_shard_${2}.done" ]
}

worker_alive() {
  local pid_file=$WORKERS/shard_${1}.pid
  [ -s "$pid_file" ] || return 1
  local pid
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null
}

start_worker() {
  local shard=$1
  local gpu=${GPU_IDS[$shard]}
  log "GPU ${gpu}, shard ${shard}의 worker가 없어 자동 재개합니다."
  (
    if ! detector_done binoculars "$shard"; then
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" score \
        --detector binoculars --shard "$shard" --device cuda:0 || exit 1
    fi
    if ! detector_done fastdetectgpt "$shard"; then
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" score \
        --detector fastdetectgpt --shard "$shard" --device cuda:0
    fi
  ) >> "$WORKERS/recovery_shard_${shard}.log" 2>&1 &
  echo "$!" > "$WORKERS/shard_${shard}.pid"
}

all_done() {
  local detector shard
  for detector in binoculars fastdetectgpt; do
    for shard in $(seq 0 $((SHARDS - 1))); do
      detector_done "$detector" "$shard" || return 1
    done
  done
}

validate() {
  "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" status > "$WORK_DIR/final_status.json" || return 1
  "$PYTHON" - "$WORK_DIR/final_status.json" "$SHARDS" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
shards = int(sys.argv[2])
expected = status["expected"]
for detector in ("binoculars", "fastdetectgpt"):
    item = status["detectors"][detector]
    assert item["scored"] == expected, (detector, item["scored"], expected)
    assert item["done_shards"] == shards, (detector, item["done_shards"], shards)
PY
}

log "watchdog을 시작합니다. poll=${POLL_SECONDS}s"
while true; do
  if all_done; then
    if validate; then
      echo "DONE $(stamp) scores=verified" | tee "$STATUS"
      log "모든 shard의 점수 수와 완료 마커를 검증했습니다."
      exit 0
    fi
    echo "RETRYING_VALIDATION $(stamp)" > "$STATUS"
    sleep "$POLL_SECONDS"
    continue
  fi
  for shard in $(seq 0 $((SHARDS - 1))); do
    if detector_done binoculars "$shard" && detector_done fastdetectgpt "$shard"; then
      continue
    fi
    worker_alive "$shard" || start_worker "$shard"
  done
  "$PYTHON" "$SCORER" --work-dir "$WORK_DIR" status > "$WORK_DIR/watch_status.json" 2>/dev/null || true
  echo "WATCHING $(stamp)" > "$STATUS"
  sleep "$POLL_SECONDS"
done
