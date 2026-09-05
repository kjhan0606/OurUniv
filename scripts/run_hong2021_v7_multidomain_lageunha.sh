#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
derived=$simba/derived/hong2021_v1
parent=$tng/training/tng100_v6_edm_laplacian_sigma2/minimum_validation.pt
out=$tng/training/tng100_simba_v7_multidomain_edm

cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_residual_v7_multidomain.py \
    --initialize "$parent" \
    --tng-train-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
    --tng-train-cache "$tng/derived/hong2021_v6/tng100_train_laplacian_sigma2.h5" \
    --simba-train-data "$derived/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$derived/simba_cv16_23_train_laplacian_sigma2.h5" \
    --tng-validation-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
    --tng-validation-cache "$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5" \
    --simba-validation-data "$derived/simba_cv24_26_validation_all_observers.h5" \
    --simba-validation-cache "$derived/simba_cv24_26_validation_laplacian_sigma2.h5" \
    --out "$out" --steps 10000 --batch 6 --validation-batch 6 \
    --workers 1 --lr 5e-5 --min-lr 5e-6 --validation-every 500 \
    --seed 3021 --device cuda
