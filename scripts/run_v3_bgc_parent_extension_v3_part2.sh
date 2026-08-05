#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
OUT=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3
GATE_OUT=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_gate_v3
P1_OUT=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v3
TAG=v3_bgc_parent_extension_v3
TEST_MANIFEST=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_n192_validation_v1/manifest_v3_bgc_validation_n192.json
MANIFEST=$OUT/manifest_${TAG}.json
PART2_MANIFEST=$OUT/manifest_${TAG}_part2.json
SEEDS=$(seq -s, 3329 3448)

mkdir -p "$OUT" "$GATE_OUT" "$P1_OUT"
"$PYTHON" "$ROOT/src/cf4_linear_cr.py" \
  --catalog "$ROOT/data/cf4_clean.npz" \
  --outdir "$OUT" --tag "$TAG" --N 192 --box-size 384 \
  --velocity-estimator bgc --cz-min 1500 --cz-max 18000 \
  --bgc-window 801 --bgc-cz-min 1500 --bgc-cz-max 18000 \
  --bgc-pool-cz-min 500 --bgc-pool-cz-max 30000 \
  --error-scale 0.9 --sigma-nl 0 --h0-prior 3 --holdout 0 \
  --sample-seeds "$SEEDS" --cg-tol 3e-5 --cg-maxiter 500

cp "$MANIFEST" "$PART2_MANIFEST"
"$PYTHON" "$ROOT/src/cf4_merge_streamed_ensemble.py" \
  --base-manifest "$PART2_MANIFEST" --outdir "$OUT" --tag "$TAG" \
  --seed-start 3193 --seed-end 3448 \
  --log "$OUT/chain_part1.log" --log "$OUT/chain_part2.log" \
  --out "$MANIFEST"

"$PYTHON" "$ROOT/src/cf4_cr_gate.py" \
  --test-manifest "$TEST_MANIFEST" --parent-manifest "$MANIFEST" \
  --outdir "$GATE_OUT"

"$PYTHON" "$ROOT/src/cf4_parent_p1.py" \
  --manifest "$MANIFEST" --config "$ROOT/config/p1_targets_v2_observer.json" \
  --outdir "$P1_OUT"
