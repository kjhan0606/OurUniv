#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v34_oriented_nonlinear_sufficiency/audit.json
log=${out%.json}.log
mkdir -p "$(dirname "$out")"
cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != lageunha ]]; then
    echo "V34 nonlinear sufficiency audit is frozen on lageunha" >&2
    exit 1
fi

python -u src/hong2021_v34_nonlinear_sufficiency.py \
    --program config/hong2021_v34_oriented_nonlinear_sufficiency_program.json \
    --repo "$repo" --out "$out" >"$log" 2>&1
