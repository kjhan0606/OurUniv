#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v46_mixture_tail_occupancy_audit
program=$repo/config/hong2021_v46_mixture_tail_occupancy_audit_program.json
audit=$root/audit.json

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V46 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V46 requires a clean worktree" >&2
  exit 1
}
[[ ! -e $root ]] || {
  echo "V46 refuses existing output: $root" >&2
  exit 1
}

mkdir -p "$root"
status=$root/status
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$root/pytest.log" 2>&1

printf "%s\n" auditing >"$status"
python -u src/hong2021_v46_tail_occupancy_audit.py \
  --program "$program" --repo "$repo" --out "$audit" \
  >"$root/audit.log" 2>&1
