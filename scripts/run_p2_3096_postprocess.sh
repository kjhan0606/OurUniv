#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
PARENT=${PARENT:-3096}
P2_DIR=${P2_DIR:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p2_n576_parent${PARENT}_v1}
P3_DIR=${P3_DIR:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p3_parent${PARENT}_v1}
RESULT=$P2_DIR/p2_screen_result.json
P1=${P1:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v2/p1_result.json}
CONFIG=${CONFIG:-$ROOT/config/p2_lg_targets_v8_bgc_n576_parent3096.json}

mkdir -p "$P3_DIR"
while [[ ! -s "$RESULT" ]]; do
  sleep 30
done

if "$PYTHON" - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
if result.get("status") != "complete":
    raise SystemExit(2)
passing = [row for row in result["results"] if row["screen_pass"]]
print(f"[post-P2] passing candidates={len(passing)}", flush=True)
raise SystemExit(0 if passing else 3)
PY
then
  :
else
  status=$?
  if [[ $status -eq 3 ]]; then
    echo "[post-P2] no candidate promoted; traceback and mask generation skipped"
    exit 0
  fi
  exit "$status"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.98}
export PYTHONUNBUFFERED=1

"$PYTHON" "$ROOT/src/cf4_p2_trace.py" \
  --p2-result "$RESULT" \
  --p1-result "$P1" \
  --config "$CONFIG" \
  --radius-mpc-h 5.0 \
  --out "$P3_DIR/lg_trace_best.npz"

"$PYTHON" "$ROOT/src/cf4_lagrangian_mask.py" \
  --input "$P3_DIR/lg_trace_best.npz" \
  --key lagrangian \
  --out "$P3_DIR/lg_mask_l9_buffer1p5.npz" \
  --box-mpc-h 384 \
  --base-level 9 \
  --buffer-mpc-h 1.5 \
  --subbox-pad-base-cells 2

echo "[post-P2] traceback and sparse L9 mask complete"
