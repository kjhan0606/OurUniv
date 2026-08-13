#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v84c0_unique_cell_tail_audit_program.json
program_sha=8ffd8ec74ee0494c195c114069cc1eabccbce50f7d83091f2c3477f372a43489
freeze_commit=0f958c453a42a135d5063d2276ffb37022b77116
run_root=$base/tng100_simba_swift_v84c0_unique_cell_tail_audit
report=$run_root/report.json
status=$run_root/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
git merge-base --is-ancestor "$freeze_commit" HEAD
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V84C0 frozen program differs" >&2
  exit 1
}
[[ ! -e $run_root ]] || {
  echo "V84C0 refuses existing output: $run_root" >&2
  exit 1
}

mkdir -p "$run_root"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V84C0_no_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V84C0 >"$status"
taskset -c 64-79 nice -n 10 pytest -q >"$run_root/pytest.log" 2>&1

printf '%s\n' auditing_inner_only_unique_cell_tail_shape >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84c0_unique_cell_tail_audit.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$run_root/audit.log" 2>&1

[[ -s $report ]]
printf '%s\n' complete_V84C0_waiting_result_record_and_review_before_any_training >"$status"
trap - EXIT
