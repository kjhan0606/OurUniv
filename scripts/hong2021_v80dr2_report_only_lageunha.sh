#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v80dr2_report_only_program.json
program_sha=cc84825d29bf969cce8bba5355356c0da78e6fe2e41198ba8329bc621f2e9db6
program_freeze=508b62608626225c9997d952740e1c1d72da1b13
sequence=$base/tng100_simba_swift_v80dr2_report_only_sequence
report_root=$base/tng100_simba_swift_v80dr2_report_only
report=$report_root/diagnostic_report.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
git merge-base --is-ancestor "$program_freeze" HEAD
[[ ! -e $sequence && ! -e $report_root ]] || {
  echo "V80DR2 refuses existing report-only output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V80DR2_no_additional_report_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V80DR2_report_only >"$status"
taskset -c 64-95 nice -n 10 pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' computing_single_non_independent_V80DR2_formula_report >"$status"
mkdir -p "$report_root"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v80dr2_report_only.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$sequence/report.log" 2>&1

if [[ $(jq -r '.frozen_V79_formula_diagnostic.would_pass_formula_if_prospective' "$report") == true ]]; then
  printf '%s\n' complete_V80DR2_promising_not_V79_pass_audit_V81_inputs >"$status"
else
  printf '%s\n' complete_V80DR2_not_promising_stop_model_path >"$status"
fi
trap - EXIT
