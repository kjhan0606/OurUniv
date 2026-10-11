#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v72_locked_spatial_quantile_transport_program.json
program_sha=83add01ae90cff6c3e1f656901daf7b9ed09a24c2f8bc59b6c52603531d8f9e4
gate_recovery_program=$repo/config/hong2021_v72_stage_A_gate_label_recovery_program.json
gate_recovery_program_sha=cbbcc1d73e6f01e2e66772e55af31181b6e4355aed5974e8f9aa837f4ff34352
sequence=$tng/evaluation/tng100_simba_swift_v72_sqt_sequence
stage_A=$tng/evaluation/tng100_simba_swift_v72_sqt_stage_A
stage_B=$tng/evaluation/tng100_simba_swift_v72_sqt_stage_B
preflight=$sequence/preflight.json
metadata_recovery=$sequence/stage_A_metadata_recovery.json
gate_recovery=$sequence/stage_A_gate_label_recovery.json
status=$sequence/status
sealed=$sequence/sealed_result.json

cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(sha256sum "$gate_recovery_program" | awk '{print $1}') == "$gate_recovery_program_sha" ]] || exit 1
[[ $(cat "$status") == 'failed_V72_metadata_recovery_resume exit=1 previous=gating_single_V72_SQT_stage_A' ]] || exit 1
[[ -d $stage_A && ! -e $stage_A/decision.json && ! -e $stage_B && ! -e $sealed ]] || exit 1
[[ $(find "$stage_A" -name metrics.json | wc -l) == 9 ]] || exit 1
[[ -e $metadata_recovery && ! -e $gate_recovery ]] || exit 1

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V72_gate_label_recovery_resume exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing_before_V72_gate_label_recovery >"$status.testing"
pytest -q >"$sequence/gate_label_recovery_pytest.log" 2>&1
rm -f "$status.testing"

# This authorization verifies the exact failure state and all eighteen frozen
# artifact hashes before the status file or any decision is changed.
python -u src/hong2021_v72_gate_label_recovery.py \
  --recovery-program "$gate_recovery_program" --v72-program "$program" \
  --repo "$repo" --out "$gate_recovery" \
  >"$sequence/gate_label_recovery.log" 2>&1

preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
printf "%s\n" gating_single_V72_SQT_stage_A_after_label_recovery >"$status"
python -u src/hong2021_v72_gate.py \
  --root "$stage_A" --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --stage A --out "$stage_A/decision.json" \
  >"$stage_A/decision_recovery.log" 2>&1

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
    --stage-B-decision "$stage_B/decision.json" \
    --metadata-recovery "$metadata_recovery" \
    --gate-label-recovery "$gate_recovery" \
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
    --metadata-recovery "$metadata_recovery" \
    --gate-label-recovery "$gate_recovery" \
    --out "$sealed" >"$sequence/seal.log" 2>&1
  printf "%s\n" complete_V72_stage_A_failure_stage_B_unopened_independent_gate_locked >"$status"
fi
trap - EXIT
