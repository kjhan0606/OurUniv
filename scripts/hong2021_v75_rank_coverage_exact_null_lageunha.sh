#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v75_rank_coverage_exact_null_audit_program.json
program_sha=6f0eaf9d06c151e429e7d378dc4f1c4460d1ea170214367fce0262ca559a13b6
root=$tng/evaluation/tng100_simba_swift_v75_rank_coverage_exact_null
sequence=$tng/evaluation/tng100_simba_swift_v75_rank_coverage_exact_null_sequence
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
    printf "failed_V75_rank_coverage_exact_null exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q tests/test_hong2021_v75_rank_coverage_exact_null.py >"$sequence/pytest.log" 2>&1
printf "%s\n" running_V75_exact_label_null_CPU64_nice15 >"$status"
taskset -c 64 nice -n 15 python -u src/hong2021_v75_rank_coverage_exact_null.py \
  --program "$program" \
  --repo "$repo" \
  --output-root "$root" >"$sequence/audit.log" 2>&1
printf "%s\n" complete_V75_rank_coverage_exact_label_null_audit >"$status"
trap - EXIT
