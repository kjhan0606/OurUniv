#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_locked_development_program.json
program_sha=5417fceb29b42108b3f75cc00f0c2c3d9a8f3cc1977b778dbb9386cce1caa7fd
train_gate=$tng/evaluation/tng100_simba_swift_v70_train_joint_structure_gate/decision.json
development=$tng/evaluation/tng100_simba_swift_v70_development
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
status=$sequence/status
sealed=$sequence/sealed_result.json

cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(cat "$status") == complete_V70_train_only_gate_pass_locked_development_authorized ]] || {
  echo "V70 development requires the sealed train-only gate pass" >&2
  exit 1
}
[[ -f $train_gate ]] || exit 1
train_gate_sha=$(sha256sum "$train_gate" | awk '{print $1}')
[[ ! -e $development ]] || {
  echo "V70 refuses an existing development output" >&2
  exit 1
}

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V70_locked_development exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing_before_V70_locked_development >"$status"
pytest -q >"$sequence/development_pytest.log" 2>&1

printf "%s\n" sampling_single_V70_locked_development >"$status"
python -u src/hong2021_v70_development_sample.py \
  --program "$program" --repo "$repo" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development" >"$sequence/development_sample.log" 2>&1

printf "%s\n" evaluating_single_V70_locked_development >"$status"
for arm in query_aligned_latent_spatial_score independent_voxel_V63_marginal; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    python -u src/hong2021_v70_development_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" gating_single_V70_locked_development >"$status"
python -u src/hong2021_v70_development_gate.py \
  --root "$development" --program "$program" --repo "$repo" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development/development_decision.json" \
  >"$development/development_decision.log" 2>&1
python -u src/hong2021_v70_seal.py \
  --program "$program" --repo "$repo" \
  --train-gate "$train_gate" \
  --development-decision "$development/development_decision.json" \
  --out "$sealed" >"$sequence/seal.log" 2>&1
if [[ $(jq -r '.development_pass' "$development/development_decision.json") == true ]]; then
  printf "%s\n" complete_V70_development_pass_waiting_explicit_EAGLE_approval >"$status"
else
  printf "%s\n" complete_V70_development_failure_independent_gate_locked >"$status"
fi
trap - EXIT
