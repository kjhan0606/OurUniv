#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
OUT=${OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_n192_validation_v1}
mkdir -p "$OUT"

"$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
  --catalog "$ROOT/data/cf4_clean.npz" \
  --outdir "$OUT" \
  --tag v3_bgc_validation_n192 \
  --N 192 \
  --box-size 384 \
  --velocity-estimator bgc \
  --cz-min 1500 \
  --cz-max 18000 \
  --bgc-window 801 \
  --bgc-cz-min 1500 \
  --bgc-cz-max 18000 \
  --bgc-pool-cz-min 500 \
  --bgc-pool-cz-max 30000 \
  --error-scale 0.9 \
  --sigma-nl 0 \
  --h0-prior 3 \
  --holdout 0.2 \
  --holdout-by-raw-index-hash \
  --split-seed 20260829 \
  --sample-seeds 201,202,203,204,205,206,207,208 \
  --cg-tol 3e-5 \
  --cg-maxiter 500 \
  2>&1 | tee "$OUT/run.log"

"$PYTHON" "$ROOT/src/cf4_bgc_n192_eval.py" \
  --config "$ROOT/config/v3_bgc_n192_validation_v1.json" \
  --manifest "$OUT/manifest_v3_bgc_validation_n192.json" \
  --outdir "$OUT"
