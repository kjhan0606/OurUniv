#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v74_query_count_energy_gate_redesign_program.json
program_sha=7b08cf433396b673430909dd8caa676da669e7ef61c8dab39d6ae9d0037850fb
root=$tng/evaluation/tng100_simba_swift_v74_gate_redesign
sequence=$tng/evaluation/tng100_simba_swift_v74_gate_redesign_sequence
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
    printf "failed_V74_gate_redesign exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q tests/test_hong2021_v74_gate_redesign.py >"$sequence/pytest.log" 2>&1
printf "%s\n" running_V74_calibration_and_verification_CPU64_nice15 >"$status"
taskset -c 64 nice -n 15 python -u src/hong2021_v74_gate_redesign.py \
  --program "$program" \
  --repo "$repo" \
  --output-root "$root" >"$sequence/audit.log" 2>&1
printf "%s\n" complete_V74_query_count_and_energy_gate_redesign >"$status"
trap - EXIT
