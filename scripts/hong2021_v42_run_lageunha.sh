#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v42_within_block_tail_body_program.json
artifact=$tng/derived/hong2021_v42/native_extreme_model.joblib
report=$tng/derived/hong2021_v42/native_extreme_model.json
sequence=$tng/evaluation/tng100_simba_swift_v42_tail_body_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v42_e23_within_block_tail_body

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V42 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V42 requires clean worktree" >&2
  exit 1
}
for path in "$artifact" "$report" "$sequence" "$evaluation"; do
  [[ ! -e $path ]] || {
    echo "V42 refuses existing output: $path" >&2
    exit 1
  }
done

mkdir -p "$sequence"
status=$sequence/status
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" fitting >"$status"
python -u src/hong2021_v42_tail_body.py fit \
  --program "$program" --repo "$repo" --artifact "$artifact" --report "$report" \
  >"$sequence/fit.log" 2>&1
artifact_sha=$(sha256sum "$artifact" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')

printf "%s\n" preflight >"$status"
python -u src/hong2021_v42_tail_body.py preflight \
  --program "$program" --repo "$repo" \
  --artifact "$artifact" --artifact-sha256 "$artifact_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf "%s\n" sampling >"$status"
python -u src/hong2021_v42_tail_body.py sample \
  --program "$program" --repo "$repo" \
  --artifact "$artifact" --artifact-sha256 "$artifact_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$evaluation" >"$sequence/sample.log" 2>&1

printf "%s\n" evaluating >"$status"
for arm in within_block_tail_body block_only_tail_control rolled_native_risk_control tail_calibration_disabled_control; do
  for domain in tng simba_dev swift_dev; do
    root=$evaluation/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" \
      --voxel-mpc-h .3125 >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" gating >"$status"
python -u src/hong2021_v42_development_gate.py \
  --root "$evaluation" --program "$program" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
