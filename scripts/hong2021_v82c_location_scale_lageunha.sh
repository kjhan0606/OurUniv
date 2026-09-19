#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v82c_consumed_loo_location_scale_program.json
program_sha=42711523d1aba1a4d1fc3350e84112728330ef1e110a3b251cd54a94001a88b5
program_freeze=716566a4e26cd1bea2caa6a83c6e4ea0421e7b22
sequence=$base/tng100_simba_swift_v82c_consumed_location_scale_sequence
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
  echo "V82C refuses existing output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(test -f "$status" && sed -n '1p' "$status" || true)
    printf 'failed_V82C_location_scale_control exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V82C >"$status"
taskset -c 64-95 nice -n 10 \
  pytest -q tests/test_hong2021_v82c_location_scale_control.py \
  >"$sequence/pytest.log" 2>&1

printf '%s\n' computing_consumed_LOO_location_scale_Gaussian >"$status"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v82c_location_scale_control.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$sequence/control.log" 2>&1

branch=$(jq -r '.decision.branch' "$report")
printf 'complete_V82C_%s\n' "$branch" >"$status"
sha256sum "$report" >"$sequence/report.sha256"
trap - EXIT
