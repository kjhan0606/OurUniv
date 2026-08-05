#!/usr/bin/env bash
set -euo pipefail

cd /home/kjhan/BACKUP/CF4

python_bin=/home/kjhan/miniconda3/bin/python
root=/gpfs/kjhan/IllustrisTNG/TNG100-1
data=${root}/derived/hong2021_v2/split00_l0_paper
cache=${root}/derived/hong2021_v6
training=${root}/training

common=(
  --train-data "${data}/tng100_train.h5"
  --train-cache "${cache}/tng100_train_laplacian_sigma2.h5"
  --validation-data "${data}/tng100_validation.h5"
  --validation-cache "${cache}/tng100_validation_laplacian_sigma2.h5"
  --steps 20000
  --batch 6
  --workers 1
  --base-channels 32
  --lr 2e-4
  --min-lr 2e-5
  --weight-decay 1e-4
  --ema-decay 0.999
  --validation-every 500
  --validation-seed 99173
  --seed 2021
  --device cuda
)

PYTHONPATH=src "${python_bin}" src/hong2021_residual_v6.py train \
  --method edm \
  --out "${training}/tng100_v6_edm_laplacian_sigma2" \
  "${common[@]}"

PYTHONPATH=src "${python_bin}" src/hong2021_residual_v6.py train \
  --method flow \
  --out "${training}/tng100_v6_flow_laplacian_sigma2" \
  "${common[@]}"
