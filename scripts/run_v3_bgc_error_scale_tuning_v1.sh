#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
OUT="$ROOT/recon/linear_cr/v3_bgc_error_scale_tuning_v1"
mkdir -p "$OUT"

for SCALE in 0.88 0.90 0.92 0.94 0.96; do
  TAG=${SCALE/./}
  "$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
    --catalog "$ROOT/data/cf4_clean.npz" \
    --outdir "$OUT" \
    --tag "v3_bgc_scale_${TAG}_n64" \
    --N 64 \
    --box-size 384 \
    --velocity-estimator bgc \
    --cz-min 1500 \
    --cz-max 18000 \
    --bgc-window 801 \
    --bgc-cz-min 1500 \
    --bgc-cz-max 18000 \
    --bgc-pool-cz-min 500 \
    --bgc-pool-cz-max 30000 \
    --error-scale "$SCALE" \
    --sigma-nl 0 \
    --h0-prior 3 \
    --holdout 0.2 \
    --holdout-by-raw-index-hash \
    --split-seed 20260817 \
    --sample-seeds 91,92,93,94,95,96,97,98 \
    --cg-tol 3e-5 \
    --cg-maxiter 500 \
    2>&1 | tee "$OUT/scale_${TAG}.log"
done

"$PYTHON" "$ROOT/src/cf4_bgc_scale_select.py" \
  --config "$ROOT/config/v3_bgc_error_scale_tuning_v1.json" \
  --manifest "$OUT/manifest_v3_bgc_scale_088_n64.json" \
  --manifest "$OUT/manifest_v3_bgc_scale_090_n64.json" \
  --manifest "$OUT/manifest_v3_bgc_scale_092_n64.json" \
  --manifest "$OUT/manifest_v3_bgc_scale_094_n64.json" \
  --manifest "$OUT/manifest_v3_bgc_scale_096_n64.json" \
  --manifest "$ROOT/recon/linear_cr/v3_bgc_n64_comparison_v2/manifest_v3_bgc_n64_paired.json" \
  --out "$OUT/error_scale_selection.json"
