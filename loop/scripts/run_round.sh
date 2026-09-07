#!/usr/bin/env bash
# 한 라운드 실행: ①후보 생성 ②prefs(부족 시 top-up 1회) ③DPO ④평가 ⑤탐지기 재학습+게이트
# 사용:  bash loop/scripts/run_round.sh <arm> <round>
#        EXTRA="--limit 32" PY=python3 bash loop/scripts/run_round.sh main 1   # 축소 실행
# 종료코드: 0 정상 · 3 탐지기 붕괴(게이트) · 그 외 실패
set -euo pipefail

ARM=${1:?arm (main|self_anchor|continual|latest_only)}
ROUND=${2:?round 번호(1..)}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$(command -v python3)}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
AC="$ROOT/loop/configs/arms/$ARM.yaml"
EXTRA=${EXTRA:-}
LOGDIR="$ROOT/loop/runs/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${ARM}_r${ROUND}.log"

step() {
  echo "===== [$ARM r$ROUND] $* =====" | tee -a "$LOG"
  # shellcheck disable=SC2086
  "$PY" "$@" $EXTRA 2>&1 | tee -a "$LOG"
}

step "$HERE/r_generate.py"    --arm-config "$AC" --round "$ROUND"
step "$HERE/r_build_prefs.py" --arm-config "$AC" --round "$ROUND" --force

# hard negative 부족 문서가 있으면 후보 증량 1회 (τ는 내리지 않는다 — notes/31)
NEEDS=$("$PY" - "$ROOT" "$AC" "$ROUND" <<'EOF'
import json, sys, pathlib
root, ac, r = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
sys.path.insert(0, str(root / "loop"))
from loop_lib import config as C
cfg = C.load_config(root / "loop/configs/base.yaml", ac)
p = C.round_dir(cfg, r) / "prefs_report.json"
print(len(json.loads(p.read_text(encoding="utf-8")).get("needs_more", [])) if p.exists() else 0)
EOF
)
if [ "$NEEDS" -gt 0 ]; then
  echo "hard negative 부족 문서 $NEEDS건 — 후보 증량 후 prefs 재구성" | tee -a "$LOG"
  step "$HERE/r_generate.py"    --arm-config "$AC" --round "$ROUND" --top-up
  step "$HERE/r_build_prefs.py" --arm-config "$AC" --round "$ROUND" --force
fi

step "$HERE/r_dpo.py"              --arm-config "$AC" --round "$ROUND"
step "$HERE/r_eval.py"             --arm-config "$AC" --round "$ROUND"
step "$HERE/r_retrain_detector.py" --arm-config "$AC" --round "$ROUND"

COLLAPSED=$("$PY" - "$ROOT" "$AC" "$ROUND" <<'EOF'
import json, sys, pathlib
root, ac, r = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
sys.path.insert(0, str(root / "loop"))
from loop_lib import config as C
cfg = C.load_config(root / "loop/configs/base.yaml", ac)
p = C.round_dir(cfg, r) / "detector" / "gate.json"
print(1 if json.loads(p.read_text(encoding="utf-8"))["collapsed"] else 0)
EOF
)
if [ "$COLLAPSED" = "1" ]; then
  echo "⚠ [$ARM r$ROUND] 탐지기 붕괴 — gate.json 참조. 다음 라운드 진행 금지." | tee -a "$LOG"
  exit 3
fi
echo "[$ARM r$ROUND] 완료" | tee -a "$LOG"
