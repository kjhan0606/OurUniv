#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
evaluation=$tng/evaluation/tng100_simba_swift_v18_e6_prior_matched_init
registry=config/hong2021_v18_development_program.json
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V18-E6 must run on Lageunha, not $host" >&2
  exit 1
fi

python - "$repo" "$registry" "$evaluation" <<'PY'
import sys
from pathlib import Path
from hong2021_v14_freeze import astrid_files
from hong2021_v15_development_gate import git_state
from hong2021_v18_edm import load_frozen_registry
repo, registry, evaluation = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
_, clean = git_state(repo)
if not clean:
    raise RuntimeError("V18-E6 preflight requires a clean committed worktree")
load_frozen_registry(registry, repo)
if astrid_files(Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")):
    raise RuntimeError("Astrid is not unopened at V18-E6 preflight")
allowed = {"initialization_variance_measurement.json"}
present = {path.name for path in evaluation.iterdir()} if evaluation.exists() else set()
unexpected = present - allowed
if unexpected:
    raise RuntimeError(f"V18-E6 refuses pre-existing result artifacts: {sorted(unexpected)}")
PY

sample_evaluate() {
  local step=$1 domain=$2 root=$3
  if [[ -e $root/ensemble16_steps40.h5 || -e $root/ensemble16_steps40.h5.partial || -e $root/ensemble_evaluation ]]; then
    echo "V18-E6 refuses pre-existing candidate output: $root" >&2
    exit 1
  fi
  mkdir -p "$root"
  python src/hong2021_v18_edm.py \
    --registry "$registry" --repo "$repo" --domain "$domain" --step "$step" \
    --out "$root/ensemble16_steps40.h5" --device cuda \
    >"$root/sample.log" 2>&1
  python src/hong2021_residual_evaluate.py \
    --candidate "edm=$root/ensemble16_steps40.h5" \
    --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$root/evaluate.log" 2>&1
}

for step in 5000 10000; do
  root=$evaluation/development_candidates/step_$(printf '%06d' "$step")
  sample_evaluate "$step" TNG100 "$root/tng"
  sample_evaluate "$step" SIMBA "$root/simba_dev"
  sample_evaluate "$step" Swift "$root/swift_dev"
done

python src/hong2021_v18_development_gate.py \
  --root "$evaluation/development_candidates" \
  --registry "$registry" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
