#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v19_e7_band_anchored_noise_edm
evaluation=$tng/evaluation/tng100_simba_swift_v19_e7_band_anchored_noise
registry=config/hong2021_v19_development_program.json
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V19-E7 gate must run on Lageunha, not $host" >&2
  exit 1
fi

python - "$repo" "$registry" "$training" "$evaluation" <<'PY'
import json, sys
from pathlib import Path
from hong2021_v14_freeze import astrid_files
from hong2021_v15_development_gate import git_state
from hong2021_v19_edm import load_frozen_registry
repo, registry, training, evaluation = map(Path, sys.argv[1:])
_, clean = git_state(repo)
if not clean:
    raise RuntimeError("V19-E7 gate preflight requires a clean committed worktree")
load_frozen_registry(registry, repo)
run = json.loads((training / "run.json").read_text())
if run.get("status") != "complete":
    raise RuntimeError("V19-E7 training is not complete")
if astrid_files(Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")):
    raise RuntimeError("Astrid is not unopened at V19-E7 preflight")
if evaluation.exists() and any(evaluation.iterdir()):
    raise RuntimeError(f"V19-E7 refuses pre-existing evaluation artifacts: {evaluation}")
PY

sample_evaluate() {
  local step=$1 domain=$2 root=$3
  if [[ -e $root ]]; then
    echo "V19-E7 refuses pre-existing candidate output: $root" >&2
    exit 1
  fi
  mkdir -p "$root"
  python src/hong2021_v19_edm.py sample \
    --registry "$registry" --repo "$repo" --training-root "$training" \
    --domain "$domain" --step "$step" \
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

python src/hong2021_v19_development_gate.py \
  --root "$evaluation/development_candidates" --training "$training" \
  --registry "$registry" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
