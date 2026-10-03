#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
log_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v33/logs
mkdir -p "$log_root"
cd "$repo"
export PYTHONPATH=$repo/src

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != syntax ]]; then
    echo "V33 TNG raw catalogues are frozen on syntax" >&2
    exit 1
fi

for split in train validation; do
    python -u src/hong2021_v33_kinematic_data.py \
        --program config/hong2021_v33_intrinsic_velocity_moment_program.json \
        --repo "$repo" --domain TNG100 --split "$split" \
        >"$log_root/tng100_${split}.log" 2>&1
done
