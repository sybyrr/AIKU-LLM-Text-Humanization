#!/usr/bin/env bash
# arm 하나를 라운드 START..END 까지 실행. 게이트 붕괴 시 중단(기록은 남음).
# 사용:
#   CUDA_VISIBLE_DEVICES=0 bash loop/scripts/run_arm.sh main 1 5
#   IGNORE_GATE=1 …               # 붕괴에도 계속 (붕괴 궤적 관찰용)
# 레인 배치(notes/31): 0=main · 1=self_anchor · 2=continual · 3=latest_only
set -euo pipefail
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

ARM=${1:?arm}
START=${2:-1}
END=${3:-5}
HERE=$(cd "$(dirname "$0")" && pwd)

for t in $(seq "$START" "$END"); do
  set +e
  bash "$HERE/run_round.sh" "$ARM" "$t"
  code=$?
  set -e
  if [ "$code" -eq 3 ]; then
    if [ "${IGNORE_GATE:-0}" = "1" ]; then
      echo "[arm $ARM] round $t 붕괴 — IGNORE_GATE=1 로 계속"
    else
      echo "[arm $ARM] round $t 에서 탐지기 붕괴로 중단. 그 자체가 결과다 (notes/31 판정표)."
      exit 3
    fi
  elif [ "$code" -ne 0 ]; then
    echo "[arm $ARM] round $t 실패 (exit $code) — 로그: loop/runs/logs/${ARM}_r${t}.log"
    exit "$code"
  fi
done
echo "[arm $ARM] round $START..$END 완료"
