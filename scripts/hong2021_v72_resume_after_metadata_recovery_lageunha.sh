#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v72_locked_spatial_quantile_transport_program.json
program_sha=83add01ae90cff6c3e1f656901daf7b9ed09a24c2f8bc59b6c52603531d8f9e4
recovery_program=$repo/config/hong2021_v72_stage_A_evaluator_metadata_recovery_program.json
recovery_program_sha=8ad681eb58075701ed680dada751b7e2f659d9850bdede9eab4d209fbf86a0e9
sequence=$tng/evaluation/tng100_simba_swift_v72_sqt_sequence
stage_A=$tng/evaluation/tng100_simba_swift_v72_sqt_stage_A
stage_B=$tng/evaluation/tng100_simba_swift_v72_sqt_stage_B
preflight=$sequence/preflight.json
recovery_record=$sequence/stage_A_metadata_recovery.json
status=$sequence/status
sealed=$sequence/sealed_result.json

cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(sha256sum "$recovery_program" | awk '{print $1}') == "$recovery_program_sha" ]] || exit 1
[[ $(cat "$status") == 'failed_V72_two_stage_SQT exit=1 previous=evaluating_single_V72_SQT_stage_A' ]] || exit 1
[[ -d $stage_A && ! -e $stage_A/decision.json && ! -e $stage_B && ! -e $sealed ]] || exit 1
[[ $(find "$stage_A" -name metrics.json | wc -l) == 0 ]] || exit 1

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V72_metadata_recovery_resume exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing_before_V72_metadata_recovery >"$status.testing"
pytest -q >"$sequence/recovery_pytest.log" 2>&1
rm -f "$status.testing"

# The recovery implementation independently verifies the original frozen
# failure-state string before this script changes status.
python -u src/hong2021_v72_metadata_recovery.py \
  --recovery-program "$recovery_program" --v72-program "$program" \
  --repo "$repo" --out "$recovery_record" \
  >"$sequence/metadata_recovery.log" 2>&1

evaluate_stage() {
  stage=$1
  root=$2
  printf "%s\n" "evaluating_single_V72_SQT_stage_$stage" >"$status"
  for arm in conditioning_stratified_spatial_quantile_transport raw_V70_latent_spatial_score independent_voxel_V63_marginal; do
    for domain in tng simba_dev swift_dev; do
      candidate=$root/$arm/fresh_candidate/$domain
      python -u src/hong2021_v72_evaluate.py \
        --candidate "sqt=$candidate/ensemble16.h5" \
        --out "$candidate/ensemble_evaluation" --voxel-mpc-h .3125 \
        >"$candidate/evaluate.log" 2>&1
    done
  done
}

evaluate_stage A "$stage_A"
printf "%s\n" gating_single_V72_SQT_stage_A >"$status"
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v72_gate.py \
  --root "$stage_A" --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --stage A --out "$stage_A/decision.json" \
  >"$stage_A/decision.log" 2>&1

if [[ $(jq -r '.stage_pass' "$stage_A/decision.json") == true ]]; then
  stage_A_sha=$(sha256sum "$stage_A/decision.json" | awk '{print $1}')
  printf "%s\n" sampling_single_V72_SQT_stage_B >"$status"
  python -u src/hong2021_v72_sample.py \
    --program "$program" --repo "$repo" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --stage B --stage-A-decision "$stage_A/decision.json" \
    --stage-A-decision-sha256 "$stage_A_sha" --out "$stage_B" \
    >"$sequence/stage_B_sample.log" 2>&1
  evaluate_stage B "$stage_B"
  printf "%s\n" gating_single_V72_SQT_stage_B >"$status"
  python -u src/hong2021_v72_gate.py \
    --root "$stage_B" --program "$program" --repo "$repo" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --stage B --stage-A-decision "$stage_A/decision.json" \
    --stage-A-decision-sha256 "$stage_A_sha" --out "$stage_B/decision.json" \
    >"$stage_B/decision.log" 2>&1
  printf "%s\n" sealing_V72_after_stage_B >"$status"
  python -u src/hong2021_v72_seal.py \
    --program "$program" --repo "$repo" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --stage-A-decision "$stage_A/decision.json" \
    --stage-B-decision "$stage_B/decision.json" --metadata-recovery "$recovery_record" \
    --out "$sealed" >"$sequence/seal.log" 2>&1
  if [[ $(jq -r '.stage_pass' "$stage_B/decision.json") == true ]]; then
    printf "%s\n" complete_V72_two_stage_pass_waiting_explicit_EAGLE_approval >"$status"
  else
    printf "%s\n" complete_V72_stage_B_failure_independent_gate_locked >"$status"
  fi
else
  printf "%s\n" sealing_V72_after_stage_A_failure >"$status"
  python -u src/hong2021_v72_seal.py \
    --program "$program" --repo "$repo" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --stage-A-decision "$stage_A/decision.json" \
    --metadata-recovery "$recovery_record" \
    --out "$sealed" >"$sequence/seal.log" 2>&1
  printf "%s\n" complete_V72_stage_A_failure_stage_B_unopened_independent_gate_locked >"$status"
fi
trap - EXIT
