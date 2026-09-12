#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v80dr_metadata_only_recovery_program.json
program_sha=b36fd4db6e5a9fbe89fecd36cf17caeb3102fc698cd053c894a7da0a160a1b5a
program_freeze=38e75b8ba7079aae9e7aabe616f8ca1efc09a325
sequence=$base/tng100_simba_swift_v80dr_metadata_recovery_sequence
ensemble_root=$base/tng100_simba_swift_v80dr_metadata_recovery_ensembles
report_root=$base/tng100_simba_swift_v80dr_engineering_report
recovery_record=$sequence/recovery_record.json
report=$report_root/diagnostic_report.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
git merge-base --is-ancestor "$program_freeze" HEAD
[[ ! -e $sequence && ! -e $ensemble_root && ! -e $ensemble_root.partial && ! -e $report_root ]] || {
  echo "V80DR refuses existing recovery output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V80DR_no_additional_repair_evaluator_or_report_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V80DR >"$status"
taskset -c 64-95 nice -n 10 pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' copying_and_proving_metadata_only_V80DR_recovery >"$status"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v80dr_metadata_recovery.py \
  --program "$program" --repo "$repo" --out "$recovery_record" \
  >"$sequence/recovery.log" 2>&1

printf '%s\n' evaluating_six_recovered_V80DR_ensembles_once >"$status"
pids=()
slot=0
for arm in candidate control; do
  for domain in tng simba_dev swift_dev; do
    root=$ensemble_root/$arm/$domain
    low=$((64 + slot * 10))
    high=$((low + 9))
    mkdir -p "$root/ensemble_evaluation"
    taskset -c "$low-$high" nice -n 10 \
      python -u src/hong2021_v80_evaluate.py \
      --candidate "v80=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1 &
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
[[ $(find "$ensemble_root" -path '*/ensemble_evaluation/metrics.json' | wc -l) == 6 ]]

printf '%s\n' computing_non_independent_recovered_V80DR_formula_diagnostic >"$status"
mkdir -p "$report_root"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v80dr_engineering_report.py \
  --program "$program" --recovery-record "$recovery_record" \
  --repo "$repo" --out "$report" \
  >"$sequence/report.log" 2>&1

if [[ $(jq -r '.frozen_V79_formula_diagnostic.would_pass_formula_if_prospective' "$report") == true ]]; then
  printf '%s\n' complete_V80DR_promising_not_V79_pass_audit_V81_inputs >"$status"
else
  printf '%s\n' complete_V80DR_not_promising_stop_model_path >"$status"
fi
trap - EXIT
