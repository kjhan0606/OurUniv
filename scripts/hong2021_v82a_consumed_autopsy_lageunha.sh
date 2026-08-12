#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v82a_consumed_rank_phase_autopsy_program.json
program_sha=7f29f2c454573fb3a622123ba9abcf4c7c999807a2ac9d79ef1833d751003e78
program_freeze=2ef7a625b11ebde7644da485c782ae6090248ae3
sequence=$base/tng100_simba_swift_v82a2_consumed_autopsy_sequence
report=$sequence/report.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
git merge-base --is-ancestor "$program_freeze" HEAD
[[ ! -e $sequence ]] || {
  echo "V82A refuses existing output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(test -f "$status" && sed -n '1p' "$status" || true)
    printf 'failed_V82A_consumed_autopsy exit=%s previous=%s\n' "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V82A >"$status"
taskset -c 64-95 nice -n 10 \
  pytest -q tests/test_hong2021_v82a_consumed_autopsy.py \
  >"$sequence/pytest.log" 2>&1

printf '%s\n' verifying_hashes_and_computing_consumed_only_V82A >"$status"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v82a_consumed_autopsy.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$sequence/autopsy.log" 2>&1

printf '%s\n' complete_V82A_consumed_only_autopsy_ready_for_Gaussian_control_design >"$status"
sha256sum "$report" >"$sequence/report.sha256"
trap - EXIT
