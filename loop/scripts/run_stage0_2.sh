#!/usr/bin/env bash
# 루프 착수 준비: 환경 확인 → D₀/앵커 → D_pair → SFT.
# 라운드는 이 뒤에 run_arm.sh 로 arm 별 실행.
# 사용:  bash loop/scripts/run_stage0_2.sh
#        EXTRA="--limit 64" bash loop/scripts/run_stage0_2.sh   # 축소 리허설
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$(command -v python3)}
EXTRA=${EXTRA:-}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

echo "===== check_env ====="
"$PY" "$ROOT/loop/check_env.py" || { echo "환경 검사 실패 — 중단"; exit 1; }

set -x
# shellcheck disable=SC2086
"$PY" "$HERE/s0_train_detector.py" $EXTRA
"$PY" "$HERE/s1_build_dpair.py" $EXTRA
"$PY" "$HERE/s2_sft.py" $EXTRA
set +x
echo "Stage 0–2 완료. 육안 검수: loop/runs/stage2/samples_dev.md"
echo "SFT 기준선 평가(선택): $PY $HERE/r_eval.py --round 0"
echo "라운드 착수: CUDA_VISIBLE_DEVICES=0 bash $HERE/run_arm.sh main 1 5"
