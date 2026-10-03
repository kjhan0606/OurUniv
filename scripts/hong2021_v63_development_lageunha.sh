#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v63_conditional_moment_model_program.json
program_sha=ea41d61a2961b3f436ed69662dc39ad8ad151980aca32863c0442948d31b6a48
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153
thresholds=$tng/evaluation/tng100_simba_swift_v54_tail_threshold_selection/thresholds.json
thresholds_sha=e5c1cd480ac47b52d568e1f8c8b8386a68230150e7f2079ad3e832220557d103
grid=$tng/evaluation/tng100_simba_swift_v56_survival_grid/grid.json
grid_sha=ba0cadb1c921c73918fcf139f121f1d9fa35e0c673ab8f62483315485bfd5fde
sequence=$tng/evaluation/tng100_simba_swift_v63_conditional_moment_sequence
status=$sequence/status
preflight=$sequence/preflight.json
preflight_sha=00d7e8fd1ad182645597d52db10b773c151d2e54669775090f55c92dcc76d4db
checkpoint=$tng/derived/hong2021_v63/conditional_moment_step12000.pt
checkpoint_sha=25beb6003ec0278bd09db184e8cda581b98da317bae91f95b4e3029ae409ac4d
report=$tng/derived/hong2021_v63/conditional_moment_step12000.json
report_sha=95332273aee3eb7838de859db61f8563b35eed502a5e5088ece4ba86104eb5c6
train_gate=$tng/evaluation/tng100_simba_swift_v63_train_high_backbone_gate/decision.json
train_gate_sha=17cddbc731f9c34b6a471709eb9a28b43e6a5e07b59e063c7ccdd082b1e0f95c
development=$tng/evaluation/tng100_simba_swift_v63_e31_conditional_moment

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V63 development requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V63 development requires clean worktree" >&2
  exit 1
}
[[ $(cat "$status") == complete_train_gate_pass_waiting_locked_development ]] || {
  echo "V63 development requires the sealed train-gate pass" >&2
  exit 1
}
for binding in \
  "$program:$program_sha" \
  "$cache:$cache_sha" \
  "$thresholds:$thresholds_sha" \
  "$grid:$grid_sha" \
  "$preflight:$preflight_sha" \
  "$checkpoint:$checkpoint_sha" \
  "$report:$report_sha" \
  "$train_gate:$train_gate_sha"; do
  path=${binding%:*}
  expected=${binding##*:}
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    echo "V63 development frozen input hash differs: $path" >&2
    exit 1
  }
done
[[ ! -e $development ]] || {
  echo "V63 refuses existing development output" >&2
  exit 1
}

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_development_sequence exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" development_testing >"$status"
pytest -q >"$sequence/development_pytest.log" 2>&1

printf "%s\n" development_sampling >"$status"
python -u src/hong2021_v63_sample.py \
  --program "$program" --repo "$repo" \
  --cache "$cache" --cache-sha256 "$cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$thresholds_sha" \
  --grid "$grid" --grid-sha256 "$grid_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development" >"$sequence/development_sample.log" 2>&1

printf "%s\n" development_evaluating >"$status"
for arm in bounded_query_local_mixture_copula rolled_parameter_control; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" development_gating >"$status"
python -u src/hong2021_v63_development_gate.py \
  --root "$development" --program "$program" --repo "$repo" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development/development_decision.json" \
  >"$development/development_decision.log" 2>&1
if [[ $(jq -r '.development_pass' "$development/development_decision.json") == true ]]; then
  printf "%s\n" complete_development_gate_pass_waiting_explicit_EAGLE_approval >"$status"
else
  printf "%s\n" complete_development_gate_failure_independent_gate_locked >"$status"
fi
trap - EXIT
