#!/usr/bin/env bash
set -euo pipefail

REPO=/home/kjhan/BACKUP/CF4
OUT=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_swift_v21_e9_conditional_affine_edm

cd "$REPO"
exec python -u src/hong2021_v21_edm.py train \
  --registry "$REPO/config/hong2021_v21_development_program.json" \
  --artifacts "$REPO/config/hong2021_v21_derived_artifacts.json" \
  --repo "$REPO" \
  --out "$OUT" \
  --device cuda
