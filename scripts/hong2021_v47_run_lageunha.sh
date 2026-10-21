#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v47_physical_moment_existence_audit
program=$repo/config/hong2021_v47_physical_moment_existence_audit_program.json

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ ! -e $root ]] || exit 1
mkdir -p "$root"
status=$root/status
trap 'code=$?; if [[ $code -eq 0 ]]; then echo complete >"$status"; else echo "failed exit=$code" >"$status"; fi' EXIT
echo testing >"$status"
pytest -q >"$root/pytest.log" 2>&1
echo auditing >"$status"
python -u src/hong2021_v47_physical_moment_audit.py \
  --program "$program" --repo "$repo" --out "$root/audit.json" \
  >"$root/audit.log" 2>&1
