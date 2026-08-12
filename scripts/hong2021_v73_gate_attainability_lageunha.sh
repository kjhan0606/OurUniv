#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v73_train_truth_gate_attainability_audit_program.json
program_sha=cf92b53504c6501faf9d6043661f070a3e7458dc57ccd94bff77365760a1cf05
root=$tng/evaluation/tng100_simba_swift_v73_gate_attainability
sequence=$tng/evaluation/tng100_simba_swift_v73_gate_attainability_sequence
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
    printf "failed_V73_gate_attainability exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q tests/test_hong2021_v73_gate_attainability.py >"$sequence/pytest.log" 2>&1
printf "%s\n" building_train_truth_summary_CPU_only_workers4_nice15_cpus64-67 >"$status"
taskset -c 64-67 nice -n 15 python -u src/hong2021_v73_gate_attainability.py \
  --program "$program" \
  --repo "$repo" \
  --output-root "$root" \
  --workers 4 >"$sequence/audit.log" 2>&1
printf "%s\n" complete_train_truth_gate_attainability_audit >"$status"
trap - EXIT
