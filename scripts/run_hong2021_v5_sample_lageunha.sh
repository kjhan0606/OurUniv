#!/usr/bin/env bash
set -euo pipefail

cd /home/kjhan/BACKUP/CF4

root=/gpfs/kjhan/IllustrisTNG/TNG100-1
data=${root}/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5
cache=${root}/derived/hong2021_v5/tng100_validation_residual_k2_4.h5
training=${root}/training/tng100_v5_stochastic_residual
evaluation=${root}/evaluation/tng100_v5_stochastic_residual
indices=49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62

mkdir -p "${evaluation}"

PYTHONPATH=src python src/hong2021_residual_diffusion.py sample \
  --data "${data}" \
  --cache "${cache}" \
  --checkpoint "${training}/minimum_validation_loss.pt" \
  --out "${evaluation}/epoch021_representative16_ensemble16.h5" \
  --indices "${indices}" \
  --ensemble 16 \
  --seed 777 \
  --device cuda \
  --voxel-mpc-h 0.3125 \
  --x0-clip 8

PYTHONPATH=src python src/hong2021_residual_diffusion.py sample \
  --data "${data}" \
  --cache "${cache}" \
  --checkpoint "${training}/last_epoch.pt" \
  --out "${evaluation}/epoch050_representative16_ensemble16.h5" \
  --indices "${indices}" \
  --ensemble 16 \
  --seed 777 \
  --device cuda \
  --voxel-mpc-h 0.3125 \
  --x0-clip 8
