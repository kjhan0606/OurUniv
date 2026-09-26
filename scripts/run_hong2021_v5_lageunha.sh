#!/usr/bin/env bash
set -euo pipefail

cd /home/kjhan/BACKUP/CF4

data_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/derived
train_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/training
v2_root=${data_root}/hong2021_v2/split00_l0_paper
v5_root=${data_root}/hong2021_v5
checkpoint=${train_root}/tng100_v4_split00_l0_groupnorm_std_cosine/minimum_validation_loss.pt

mkdir -p "${v5_root}"

PYTHONPATH=src python src/hong2021_residual_diffusion.py prepare \
  --data "${v2_root}/tng100_train.h5" \
  --checkpoint "${checkpoint}" \
  --out "${v5_root}/tng100_train_residual_k2_4.h5" \
  --batch 6 \
  --workers 2 \
  --device cuda \
  --k-low-h-mpc 2.0 \
  --k-high-h-mpc 4.0

PYTHONPATH=src python src/hong2021_residual_diffusion.py prepare \
  --data "${v2_root}/tng100_validation.h5" \
  --checkpoint "${checkpoint}" \
  --out "${v5_root}/tng100_validation_residual_k2_4.h5" \
  --batch 6 \
  --workers 2 \
  --device cuda \
  --k-low-h-mpc 2.0 \
  --k-high-h-mpc 4.0

PYTHONPATH=src python src/hong2021_residual_diffusion.py train \
  --train-data "${v2_root}/tng100_train.h5" \
  --train-cache "${v5_root}/tng100_train_residual_k2_4.h5" \
  --validation-data "${v2_root}/tng100_validation.h5" \
  --validation-cache "${v5_root}/tng100_validation_residual_k2_4.h5" \
  --out "${train_root}/tng100_v5_stochastic_residual" \
  --epochs 50 \
  --batch 6 \
  --workers 2 \
  --base-channels 32 \
  --diffusion-steps 200 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --weight-decay 1e-4 \
  --seed 2021 \
  --device cuda
