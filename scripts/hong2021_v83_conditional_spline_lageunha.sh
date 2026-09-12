#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v83_conditional_marginal_spline_program.json
expected_program_sha=035e52b3d7059816b61dbf2b23e0cca9f5c5592903f704f0103a913556cea174
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
expected_cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153
sequence=$tng/evaluation/tng100_simba_swift_v83_conditional_marginal_spline_sequence
preflight=$sequence/preflight.json
derived=$tng/derived/hong2021_v83_conditional_marginal_spline
checkpoint=$derived/step12000.pt
training_report=$derived/training_report.json
train_gate=$tng/evaluation/tng100_simba_swift_v83_conditional_marginal_spline_train_gate/decision.json
ensemble_root=$tng/evaluation/tng100_simba_swift_v83_conditional_marginal_spline_consumed_development
metrics_root=$tng/evaluation/tng100_simba_swift_v83_conditional_marginal_spline_consumed_metrics
development_gate=$tng/evaluation/tng100_simba_swift_v83_conditional_marginal_spline_consumed_gate/decision.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V83 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V83 requires a clean frozen worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$expected_program_sha" ]] || {
  echo "V83 program hash differs" >&2
  exit 1
}
[[ $(sha256sum "$cache" | awk '{print $1}') == "$expected_cache_sha" ]] || {
  echo "V83 conditioning cache hash differs" >&2
  exit 1
}
for path in "$sequence" "$derived" "$(dirname "$train_gate")" \
  "$ensemble_root" "$metrics_root" "$(dirname "$development_gate")"; do
  [[ ! -e $path ]] || {
    echo "V83 refuses existing output: $path" >&2
    exit 1
  }
done

mkdir -p "$sequence"
record_failure() {
  code=$?
  previous=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf 'failed_V83_sequence exit=%s previous=%s\n' "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V83 >"$status"
taskset -c 64-79 nice -n 5 pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' running_train_only_hard_preflight >"$status"
taskset -c 64-79 nice -n 5 python -u src/hong2021_v83_preflight.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$expected_cache_sha" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf '%s\n' training_fixed_12000_step_V83 >"$status"
taskset -c 64-79 nice -n 5 python -u src/hong2021_v83_train.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$expected_cache_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --report "$training_report" \
  >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
training_report_sha=$(sha256sum "$training_report" | awk '{print $1}')

printf '%s\n' evaluating_sealed_train_only_holdout >"$status"
taskset -c 64-79 nice -n 5 python -u src/hong2021_v83_train_gate.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$expected_cache_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$training_report" --report-sha256 "$training_report_sha" \
  --out "$train_gate" >"$sequence/train_gate.log" 2>&1
if [[ $(jq -r '.train_holdout_mechanism_pass' "$train_gate") != true ]]; then
  printf '%s\n' complete_V83_train_holdout_failure_development_locked >"$status"
  trap - EXIT
  exit 0
fi
train_gate_sha=$(sha256sum "$train_gate" | awk '{print $1}')

printf '%s\n' sampling_one_consumed_development_candidate_and_control >"$status"
taskset -c 64-79 nice -n 5 python -u src/hong2021_v83_sample.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$expected_cache_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --output-root "$ensemble_root" >"$sequence/sample.log" 2>&1

printf '%s\n' evaluating_six_consumed_development_ensembles >"$status"
pids=()
slot=0
for arm in candidate control; do
  for domain in tng simba_dev swift_dev; do
    ensemble=$ensemble_root/$arm/$domain/ensemble16.h5
    destination=$metrics_root/$arm/$domain
    low=$((64 + slot * 10))
    high=$((low + 9))
    mkdir -p "$destination"
    taskset -c "$low-$high" nice -n 10 \
      python -u src/hong2021_v83_evaluate.py \
      --candidate "v83=$ensemble" --out "$destination" --voxel-mpc-h .3125 \
      >"$destination/evaluate.log" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

printf '%s\n' computing_consumed_only_V83_engineering_gate >"$status"
taskset -c 64-79 nice -n 5 python -u src/hong2021_v83_development_gate.py \
  --program "$program" --repo "$repo" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --ensemble-root "$ensemble_root" --metrics-root "$metrics_root" \
  --out "$development_gate" >"$sequence/development_gate.log" 2>&1
if [[ $(jq -r '.consumed_development_engineering_pass' "$development_gate") == true ]]; then
  printf '%s\n' complete_V83_consumed_gate_pass_waiting_user_approval_before_independent_validation >"$status"
else
  printf '%s\n' complete_V83_consumed_gate_failure_independent_validation_locked >"$status"
fi
trap - EXIT
