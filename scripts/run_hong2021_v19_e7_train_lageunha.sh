#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
out=$tng/training/tng100_simba_swift_v19_e7_band_anchored_noise_edm
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V19-E7 training must run on Lageunha, not $host" >&2
  exit 1
fi

python src/hong2021_v19_edm.py train \
  --registry config/hong2021_v19_development_program.json \
  --repo "$repo" --out "$out" --device cuda
