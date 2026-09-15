#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
OUT=${OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v1}
GATE_OUT=${GATE_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_gate_v1}
P1_OUT=${P1_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v1}
TEST_MANIFEST=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_n192_validation_v1/manifest_v3_bgc_validation_n192.json
TAG=${TAG:-v3_bgc_parent_extension}
SEEDS=${SEEDS:-$(seq -s, 3017 3064)}
mkdir -p "$OUT" "$GATE_OUT" "$P1_OUT"

"$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
  --catalog "$ROOT/data/cf4_clean.npz" \
  --outdir "$OUT" \
  --tag "$TAG" \
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
  --holdout 0 \
  --sample-seeds "$SEEDS" \
  --cg-tol 3e-5 \
  --cg-maxiter 500 \
  2>&1 | tee "$OUT/run.log"

"$PYTHON" "$ROOT/src/cf4_cr_gate.py" \
  --test-manifest "$TEST_MANIFEST" \
  --parent-manifest "$OUT/manifest_${TAG}.json" \
  --outdir "$GATE_OUT" \
  2>&1 | tee "$GATE_OUT/gate.log"

"$PYTHON" "$ROOT/src/cf4_parent_p1.py" \
  --manifest "$OUT/manifest_${TAG}.json" \
  --config "$ROOT/config/p1_targets_v2_observer.json" \
  --outdir "$P1_OUT" \
  2>&1 | tee "$P1_OUT/run.log"
