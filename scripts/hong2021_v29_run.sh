#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v29_physical_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v29_e17_direct_physical_residual
candidate=$evaluation/development_candidate
registry=$repo/config/hong2021_v29_development_program.json

cd "$repo"
[[ -z $(git status --porcelain) ]] || { echo "V29 requires clean worktree" >&2; exit 1; }
[[ ! -e $preflight && ! -e $evaluation ]] || { echo "V29 outputs already exist" >&2; exit 1; }
mkdir -p "$sequence" "$evaluation"
export PYTHONPATH=$repo/src
python -m pytest -q
python -u scripts/hong2021_v29_preflight.py \
  --registry "$registry" --repo "$repo" --out "$preflight" \
  >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v29_physical.py \
  --registry "$registry" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$candidate" >"$evaluation/sample.log" 2>&1
for domain in tng simba_dev swift_dev; do
  root=$candidate/$domain
  python -u src/hong2021_residual_evaluate.py \
    --candidate "edm=$root/ensemble16.h5" \
    --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$root/evaluate.log" 2>&1
done
python -u src/hong2021_v29_development_gate.py \
  --root "$candidate" --registry "$registry" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
