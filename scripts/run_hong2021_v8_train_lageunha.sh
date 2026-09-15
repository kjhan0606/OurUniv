#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_v8_observable_context

cd "$repo"
python src/hong2021_residual_v8_context.py train \
  --initialize /gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_v7_multidomain_edm/minimum_validation.pt \
  --tng-train-data /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v2/split00_l0_paper/tng100_train.h5 \
  --tng-train-cache /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v6/tng100_train_laplacian_sigma2.h5 \
  --simba-train-data /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5 \
  --simba-train-cache /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv16_23_train_laplacian_sigma2.h5 \
  --tng-validation-data /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5 \
  --tng-validation-cache /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5 \
  --simba-validation-data /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5 \
  --simba-validation-cache /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_laplacian_sigma2.h5 \
  --out "$out" \
  --steps 10000 --validation-every 500 --workers 1
