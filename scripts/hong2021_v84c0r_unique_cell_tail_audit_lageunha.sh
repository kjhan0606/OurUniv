#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v84c0_unique_cell_tail_audit_program.json
program_sha=027b054a264e73739e07ad3b0163fe35cc970843097129b4b0160148c2ba10d7
freeze_commit=75f1f1904c775cddb0c8c6ec194f70bb209add5c
run_root=$base/tng100_simba_swift_v84c0r_unique_cell_tail_audit
report=$run_root/report.json
status=$run_root/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
git merge-base --is-ancestor "$freeze_commit" HEAD
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V84C0R frozen amended program differs" >&2
  exit 1
}
[[ ! -e $run_root ]] || {
  echo "V84C0R refuses existing output: $run_root" >&2
  exit 1
}

mkdir -p "$run_root"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V84C0R_no_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V84C0R >"$status"
taskset -c 64-79 nice -n 10 pytest -q >"$run_root/pytest.log" 2>&1

printf '%s\n' auditing_inner_only_unique_cell_tail_shape >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84c0_unique_cell_tail_audit.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$run_root/audit.log" 2>&1

[[ -s $report ]]
printf '%s\n' complete_V84C0R_waiting_result_record_and_review_before_any_training >"$status"
trap - EXIT
