#!/usr/bin/env bash
set -euo pipefail

cd /home/kjhan/BACKUP/CF4

while tmux has-session -t hong2021_v5_sample 2>/dev/null; do
  sleep 30
done

root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_v5_stochastic_residual
epoch21=${root}/epoch021_representative16_ensemble16.h5
epoch50=${root}/epoch050_representative16_ensemble16.h5

test -f "${epoch21}"
test -f "${epoch50}"

PYTHONPATH=src python src/hong2021_residual_evaluate.py \
  --candidate "epoch021=${epoch21}" \
  --candidate "epoch050=${epoch50}" \
  --out "${root}/ensemble_evaluation" \
  --voxel-mpc-h 0.3125
