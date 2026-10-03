#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v69_pair_estimator_rank_convergence_program.json
program_sha=ce8d61cd4a82623b0c755c2379d33f71b08d7eea60289e833a9d54d88ac2e940
root=$tng/evaluation/tng100_simba_swift_v69_pair_estimator_rank_convergence
sequence=$tng/evaluation/tng100_simba_swift_v69_pair_estimator_rank_convergence_sequence
audit=$root/audit.json
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
for path in "$root" "$sequence"; do [[ ! -e $path ]] || exit 1; done
mkdir -p "$sequence"
record_failure() {
  code=$?; current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V69_audit exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1
printf "%s\n" auditing_no_refit_train_only_pair_estimator_rank_convergence >"$status"
python -u src/hong2021_v69_pair_estimator_rank_convergence.py \
  --program "$program" --repo "$repo" --out "$audit" >"$sequence/audit.log" 2>&1
printf "%s\n" complete_no_refit_train_only_pair_estimator_rank_convergence_audit >"$status"
trap - EXIT
