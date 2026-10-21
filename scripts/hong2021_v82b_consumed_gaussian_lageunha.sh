#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v82b_consumed_loo_gaussian_controls_program.json
program_sha=8aca170b2a8441ab6cb70329c63a094c4f3b311e38933954f236b2e7d89b04b8
program_freeze=3a3027db9ffb6928252fc7c08a6750f00f9c9410
sequence=$base/tng100_simba_swift_v82b_consumed_gaussian_sequence
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
  echo "V82B refuses existing output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(test -f "$status" && sed -n '1p' "$status" || true)
    printf 'failed_V82B_consumed_Gaussian_controls exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V82B >"$status"
taskset -c 64-95 nice -n 10 \
  pytest -q tests/test_hong2021_v82b_gaussian_control.py \
  >"$sequence/pytest.log" 2>&1

printf '%s\n' computing_consumed_LOO_Gaussian_controls >"$status"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v82b_gaussian_control.py \
  --program "$program" --repo "$repo" --out "$report" \
  >"$sequence/control.log" 2>&1

branch=$(jq -r '.decision.branch' "$report")
printf 'complete_V82B_%s\n' "$branch" >"$status"
sha256sum "$report" >"$sequence/report.sha256"
trap - EXIT
