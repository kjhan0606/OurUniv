#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v33_intrinsic_velocity_information/audit.json
log=${out%.json}.log
mkdir -p "$(dirname "$out")"
cd "$repo"
export PYTHONPATH=$repo/src

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != lageunha ]]; then
    echo "V33 information audit is frozen on lageunha" >&2
    exit 1
fi

python -u src/hong2021_v33_information_audit.py \
    --program config/hong2021_v33_intrinsic_velocity_moment_program.json \
    --repo "$repo" --out "$out" >"$log" 2>&1
