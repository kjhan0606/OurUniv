#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v68_population_pair_objective_audit_program.json
program_sha=818433d4567b67a3f9ee0eca2271d25da720cd2baf4b4ef042e96b0ad90a852c
root=$tng/evaluation/tng100_simba_swift_v68_population_pair_objective_audit
sequence=$tng/evaluation/tng100_simba_swift_v68_population_pair_objective_sequence
audit=$root/audit.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
for path in "$root" "$sequence"; do [[ ! -e $path ]] || exit 1; done
mkdir -p "$sequence"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V68_audit exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1
printf "%s\n" auditing_no_refit_train_only_population_pair_objective >"$status"
python -u src/hong2021_v68_population_pair_objective_audit.py \
  --program "$program" --repo "$repo" --out "$audit" >"$sequence/audit.log" 2>&1
printf "%s\n" complete_no_refit_train_only_population_pair_objective_audit >"$status"
trap - EXIT
