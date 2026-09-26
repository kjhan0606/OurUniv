#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
out=$tng/training/tng100_simba_swift_v20_e8_gaussianized_marginal_edm
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V20-E8 training must run on Lageunha, not $host" >&2
  exit 1
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ ${gpu_name,,} != *ada* ]]; then
  echo "V20-E8 requires the Lageunha Ada GPU, found: $gpu_name" >&2
  exit 1
fi

python src/hong2021_v20_edm.py train \
  --registry config/hong2021_v20_development_program.json \
  --repo "$repo" --out "$out" --device cuda
