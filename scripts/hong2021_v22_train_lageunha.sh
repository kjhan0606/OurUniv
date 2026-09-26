#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_swift_v22_e10_long_horizon_edm
cd "$repo"
exec python -u src/hong2021_v22_edm.py train \
  --registry "$repo/config/hong2021_v22_development_program.json" \
  --repo "$repo" --out "$out" --device cuda
