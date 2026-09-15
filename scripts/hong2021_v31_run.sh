#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
registry=$repo/config/hong2021_v31_development_program.json
model=$tng/derived/hong2021_v31/physical_conditional_copula.npz
model_report=$tng/derived/hong2021_v31/physical_conditional_copula.json
sequence=$tng/evaluation/tng100_simba_swift_v31_copula_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v31_e18_physical_conditional_copula
candidate=$evaluation/development_candidate

cd "$repo"
[[ -z $(git status --porcelain) ]] || { echo "V31 requires clean worktree" >&2; exit 1; }
for path in "$model" "$model_report" "$preflight" "$evaluation"; do
  [[ ! -e $path ]] || { echo "V31 output already exists: $path" >&2; exit 1; }
done
mkdir -p "${model%/*}" "$sequence" "$evaluation"
export PYTHONPATH=$repo/src
python -m pytest -q
python -u src/hong2021_v31_copula.py fit \
  --registry "$registry" --repo "$repo" \
  --artifact "$model" --report "$model_report" \
  >"$sequence/fit.log" 2>&1
model_sha=$(sha256sum "$model" | awk '{print $1}')
model_report_sha=$(sha256sum "$model_report" | awk '{print $1}')
python -u scripts/hong2021_v31_preflight.py \
  --registry "$registry" --repo "$repo" \
  --model "$model" --model-report "$model_report" --out "$preflight" \
  >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v31_copula.py sample \
  --registry "$registry" --repo "$repo" \
  --model "$model" --model-sha256 "$model_sha" \
  --model-report "$model_report" --model-report-sha256 "$model_report_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$candidate" >"$evaluation/sample.log" 2>&1
for domain in tng simba_dev swift_dev; do
  root=$candidate/$domain
  python -u src/hong2021_residual_evaluate.py \
    --candidate "edm=$root/ensemble16.h5" \
    --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$root/evaluate.log" 2>&1
done
python -u src/hong2021_v31_development_gate.py \
  --root "$candidate" --registry "$registry" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
