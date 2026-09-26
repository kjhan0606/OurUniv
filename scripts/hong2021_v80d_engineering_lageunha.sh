#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v80d_engineering_diagnostic_program.json
program_sha=318a01d4b28e2950624af0835836feaf9884db78a6a098b3571b114312587fc6
program_freeze=c0484cedcb0286964413cca1a291e31e8a23badf
sequence=$base/tng100_simba_swift_v80d_engineering_sequence
ensemble_root=$base/tng100_simba_swift_v80d_engineering_ensembles
report_root=$base/tng100_simba_swift_v80d_engineering_report
preflight=$sequence/preflight.json
report=$report_root/diagnostic_report.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
git merge-base --is-ancestor "$program_freeze" HEAD
[[ ! -e $sequence && ! -e $ensemble_root && ! -e $report_root ]] || {
  echo "V80D refuses existing engineering output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V80D_no_additional_fix_or_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V80D >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' running_code_only_V80D_preflight >"$status"
python -u src/hong2021_v80d_engineering_sample.py preflight \
  --program "$program" --repo "$repo" --output-root "$ensemble_root" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1

printf '%s\n' sampling_single_V80D_engineering_candidate_control >"$status"
python -u src/hong2021_v80d_engineering_sample.py sample \
  --program "$program" --preflight "$preflight" --repo "$repo" \
  --output-root "$ensemble_root" >"$sequence/sample.log" 2>&1

printf '%s\n' evaluating_six_V80D_engineering_ensembles >"$status"
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
for pid in "${pids[@]}"; do
  wait "$pid"
done

printf '%s\n' computing_non_independent_V80D_formula_diagnostic >"$status"
mkdir -p "$report_root"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v80d_engineering_report.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$sequence/report.log" 2>&1

if [[ $(jq -r '.frozen_V79_formula_diagnostic.would_pass_formula_if_prospective' "$report") == true ]]; then
  printf '%s\n' complete_V80D_promising_not_V79_pass_audit_V81_inputs >"$status"
else
  printf '%s\n' complete_V80D_not_promising_stop_model_path >"$status"
fi
trap - EXIT
