#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/IllustrisTNG/TNG100-1
data=$root/derived/hong2021_v2/split00_l0_paper
cache=$root/derived/hong2021_v6
out=$root/evaluation/hong2021_density_likelihood_e2

cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_wiener_self_consistency.py \
    --train-data "$data/tng100_train.h5" \
    --train-mean "$cache/tng100_train_laplacian_sigma2.h5" \
    --validation-data "$data/tng100_validation.h5" \
    --validation-mean "$cache/tng100_validation_laplacian_sigma2.h5" \
    --out "$out"
