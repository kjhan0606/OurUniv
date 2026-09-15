#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v76_separate_rank_coverage_nonmasking_audit_program.json
program_sha=f72e2b0ff7ae7bd23acea3e120f0fb6e89b4088c57b6b7feb1405eef973796c0
root=$tng/evaluation/tng100_simba_swift_v76_separate_rank_coverage_nonmasking
sequence=$tng/evaluation/tng100_simba_swift_v76_separate_rank_coverage_nonmasking_sequence
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
for path in "$root" "$sequence"; do
  [[ ! -e $path ]] || exit 1
done

mkdir -p "$sequence"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V76_separate_rank_coverage exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q tests/test_hong2021_v76_separate_rank_coverage_nonmasking.py >"$sequence/pytest.log" 2>&1
printf "%s\n" running_V76_separate_rank_coverage_CPU64_nice15 >"$status"
taskset -c 64 nice -n 15 python -u src/hong2021_v76_separate_rank_coverage_nonmasking.py \
  --program "$program" \
  --repo "$repo" \
  --output-root "$root" >"$sequence/audit.log" 2>&1
printf "%s\n" complete_V76_separate_rank_coverage_nonmasking_audit >"$status"
trap - EXIT
