#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
OUT="$ROOT/recon/linear_cr/v3_bgc_n64_comparison_v1"
mkdir -p "$OUT"

COMMON=(
  --catalog "$ROOT/data/cf4_clean.npz"
  --outdir "$OUT"
  --N 64
  --box-size 384
  --cz-min 1500
  --cz-max 18000
  --sigma-nl 0
  --h0-prior 3
  --holdout 0.2
  --split-seed 20260813
  --sample-seeds 81,82,83,84,85,86,87,88
  --cg-tol 3e-5
  --cg-maxiter 500
)

"$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
  "${COMMON[@]}" \
  --tag v3_bgc_control_n64 \
  --velocity-estimator wf15 \
  --error-scale 0.9 \
  2>&1 | tee "$OUT/control.log"

"$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
  "${COMMON[@]}" \
  --tag v3_bgc_n64 \
  --velocity-estimator bgc \
  --error-scale 1.0 \
  --bgc-window 801 \
  --bgc-cz-min 1500 \
  --bgc-cz-max 18000 \
  --bgc-pool-cz-min 500 \
  --bgc-pool-cz-max 30000 \
  2>&1 | tee "$OUT/bgc.log"

"$PYTHON" "$ROOT/src/cf4_bgc_cr_eval.py" \
  --config "$ROOT/config/v3_bgc_n64_comparison_v1.json" \
  --control "$OUT/manifest_v3_bgc_control_n64.json" \
  --bgc "$OUT/manifest_v3_bgc_n64.json" \
  --outdir "$OUT"
