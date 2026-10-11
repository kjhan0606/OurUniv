#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v41_two_stage_structure_amplitude_program.json
artifact=$tng/derived/hong2021_v41/two_stage_model.joblib
report=$tng/derived/hong2021_v41/two_stage_model.json
sequence=$tng/evaluation/tng100_simba_swift_v41_two_stage_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v41_e22_two_stage_structure_amplitude

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || { echo "V41 requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V41 requires clean worktree" >&2; exit 1; }
for path in "$artifact" "$report" "$sequence" "$evaluation"; do
  [[ ! -e $path ]] || { echo "V41 refuses existing output: $path" >&2; exit 1; }
done

mkdir -p "$sequence"
pytest -q >"$sequence/pytest.log" 2>&1
python -u src/hong2021_v41_two_stage.py fit \
  --program "$program" --repo "$repo" --artifact "$artifact" --report "$report" \
  >"$sequence/fit.log" 2>&1
artifact_sha=$(sha256sum "$artifact" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')
python -u src/hong2021_v41_two_stage.py preflight \
  --program "$program" --repo "$repo" \
  --artifact "$artifact" --artifact-sha256 "$artifact_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v41_two_stage.py sample \
  --program "$program" --repo "$repo" \
  --artifact "$artifact" --artifact-sha256 "$artifact_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$evaluation" >"$sequence/sample.log" 2>&1

for arm in two_stage backbone_risk_ablation rolled_risk_control shuffled_amplitude_control; do
  for domain in tng simba_dev swift_dev; do
    root=$evaluation/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" \
      --voxel-mpc-h .3125 >"$root/evaluate.log" 2>&1
  done
done

python -u src/hong2021_v41_development_gate.py \
  --root "$evaluation" --program "$program" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
