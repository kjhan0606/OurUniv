#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
base=$tng/evaluation
program=$repo/config/hong2021_v83r_metadata_only_recovery_program.json
program_sha=a4387f50d7d58de479f080b10bc2426bd1b97cde44ec83c5661591ed6f92ddf4
v83_program=$repo/config/hong2021_v83_conditional_marginal_spline_program.json
v83_program_sha=035e52b3d7059816b61dbf2b23e0cca9f5c5592903f704f0103a913556cea174
sequence=$base/tng100_simba_swift_v83r_metadata_recovery_sequence
recovery_record=$sequence/recovery_record.json
ensemble_root=$base/tng100_simba_swift_v83r_metadata_recovery_ensembles
metrics_root=$base/tng100_simba_swift_v83r_metadata_recovery_metrics
development_gate=$base/tng100_simba_swift_v83r_metadata_recovery_gate/decision.json
checkpoint=$tng/derived/hong2021_v83_conditional_marginal_spline/step12000.pt
checkpoint_sha=fc06559221e3430f95dfd3de0131e3646d364d631cab462faabec55e1eb9572d
train_gate=$base/tng100_simba_swift_v83_conditional_marginal_spline_train_gate/decision.json
train_gate_sha=d615f5b5d1a89ee8e007d3ccab1d35ec222907484814d984ce08b50c2438d032
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
for binding in \
  "$program:$program_sha" \
  "$v83_program:$v83_program_sha" \
  "$checkpoint:$checkpoint_sha" \
  "$train_gate:$train_gate_sha"; do
  path=${binding%:*}
  expected=${binding##*:}
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    echo "V83R frozen input differs: $path" >&2
    exit 1
  }
done
for path in "$sequence" "$ensemble_root" "$ensemble_root.partial" \
  "$metrics_root" "$(dirname "$development_gate")"; do
  [[ ! -e $path ]] || {
    echo "V83R refuses existing recovery output: $path" >&2
    exit 1
  }
done

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V83R_no_additional_recovery_or_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V83R >"$status"
taskset -c 64-79 nice -n 10 pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' copying_and_proving_dataset_identical_V83R_recovery >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v83r_metadata_recovery.py \
  --program "$program" --repo "$repo" --out "$recovery_record" \
  >"$sequence/recovery.log" 2>&1
recovery_record_sha=$(sha256sum "$recovery_record" | awk '{print $1}')

printf '%s\n' evaluating_six_dataset_identical_recovered_ensembles >"$status"
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
evaluation_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    evaluation_failed=1
  fi
done
[[ $evaluation_failed -eq 0 ]]
[[ $(find "$metrics_root" -name metrics.json | wc -l) == 6 ]]

printf '%s\n' computing_recovered_consumed_only_V83_gate >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v83r_development_gate.py \
  --recovery-program "$program" --v83-program "$v83_program" --repo "$repo" \
  --recovery-record "$recovery_record" \
  --recovery-record-sha256 "$recovery_record_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --ensemble-root "$ensemble_root" --metrics-root "$metrics_root" \
  --out "$development_gate" >"$sequence/development_gate.log" 2>&1

if [[ $(jq -r '.consumed_development_engineering_pass' "$development_gate") == true ]]; then
  printf '%s\n' complete_V83R_consumed_gate_pass_waiting_user_approval_before_independent_validation >"$status"
else
  printf '%s\n' complete_V83R_consumed_gate_failure_independent_validation_locked >"$status"
fi
trap - EXIT
