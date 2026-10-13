#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v21_e9_conditional_affine_edm
evaluation=$tng/evaluation/tng100_simba_swift_v21_e9_conditional_affine
registry=$repo/config/hong2021_v21_development_program.json
artifacts=$repo/config/hong2021_v21_derived_artifacts.json
cd "$repo"
export PYTHONPATH=$repo/src

if [[ ${HOSTNAME,,} != lageunha ]]; then
  echo "V21 gate must run on Lageunha" >&2
  exit 1
fi
python - "$training" "$evaluation" <<'PY'
import json, sys
from pathlib import Path
training, evaluation = map(Path, sys.argv[1:])
if json.loads((training / "run.json").read_text()).get("status") != "complete":
    raise RuntimeError("V21 training is not complete")
if evaluation.exists() and any(evaluation.iterdir()):
    raise RuntimeError(f"V21 refuses pre-existing evaluation artifacts: {evaluation}")
PY

sample_evaluate() {
  local step=$1 domain=$2 root=$3
  mkdir -p "$root"
  python src/hong2021_v21_edm.py sample \
    --registry "$registry" --artifacts "$artifacts" --repo "$repo" \
    --training-root "$training" --domain "$domain" --step "$step" \
    --out "$root/ensemble16_steps40.h5" --device cuda \
    >"$root/sample.log" 2>&1
  python src/hong2021_residual_evaluate.py \
    --candidate "edm=$root/ensemble16_steps40.h5" \
    --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$root/evaluate.log" 2>&1
}

mkdir -p "$evaluation"
for step in 5000 10000; do
  root=$evaluation/development_candidates/step_$(printf '%06d' "$step")
  sample_evaluate "$step" TNG100 "$root/tng"
  sample_evaluate "$step" SIMBA "$root/simba_dev"
  sample_evaluate "$step" Swift "$root/swift_dev"
done

python src/hong2021_v21_development_gate.py \
  --root "$evaluation/development_candidates" --training "$training" \
  --registry "$registry" --artifacts "$artifacts" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
