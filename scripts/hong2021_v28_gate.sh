#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v28_empirical_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v28_e16_empirical_joint_control
candidate=$evaluation/development_candidate
registry=$repo/config/hong2021_v28_development_program.json

[[ ${HOSTNAME,,} == lageunha ]] || {
  echo "V28 gate requires Lageunha" >&2
  exit 1
}
cd "$repo"
[[ -z $(git status --porcelain) ]] || {
  echo "V28 gate requires a clean committed worktree" >&2
  exit 1
}
[[ -s $preflight ]] || {
  echo "V28 hard preflight is absent" >&2
  exit 1
}
[[ ! -e $evaluation ]] || {
  echo "V28 evaluation root already exists" >&2
  exit 1
}
mkdir -p "$evaluation"
export PYTHONPATH=$repo/src
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v28_empirical.py sample-all \
  --registry "$registry" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$candidate" --device cuda \
  >"$evaluation/sample.log" 2>&1
for domain in tng simba_dev swift_dev; do
  root=$candidate/$domain
  python -u src/hong2021_residual_evaluate.py \
    --candidate "edm=$root/ensemble16.h5" \
    --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$root/evaluate.log" 2>&1
done
python -u src/hong2021_v28_development_gate.py \
  --root "$candidate" --registry "$registry" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
