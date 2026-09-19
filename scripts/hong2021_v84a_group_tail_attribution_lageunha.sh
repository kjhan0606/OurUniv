#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v84a_group_tail_attribution_program.json
program_sha=549439e97d5c90665d5a3f40068b5d403459bbc40b61ed8f54e8155467a322c2
freeze_commit=ec1daa3b1a06b74adbe4fe76b6e059b2efa43fb3
run_root=$base/tng100_simba_swift_v84a_group_tail_attribution
report=$run_root/report.json
status=$run_root/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
git merge-base --is-ancestor "$freeze_commit" HEAD
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V84A frozen program differs" >&2
  exit 1
}
[[ ! -e $run_root ]] || {
  echo "V84A refuses existing output: $run_root" >&2
  exit 1
}

mkdir -p "$run_root"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V84A_no_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V84A >"$status"
taskset -c 64-79 nice -n 10 pytest -q >"$run_root/pytest.log" 2>&1

printf '%s\n' auditing_group_leakage_direct_PIT_tails_and_SQT_attribution >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84a_group_tail_attribution.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$run_root/audit.log" 2>&1

[[ -s $report ]]
printf '%s\n' complete_V84A_waiting_result_record_and_review >"$status"
trap - EXIT
