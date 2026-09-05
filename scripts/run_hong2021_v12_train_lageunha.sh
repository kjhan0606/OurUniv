#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
out=$tng/training/tng100_simba_v12_gaussianized
cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_residual_v12_gaussianized.py train \
    --initialize "$tng/training/tng100_simba_v11_recentered/validation_checkpoints/step_010000.pt" \
    --tng-train-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
    --tng-train-cache "$tng/derived/hong2021_v12/tng100_train_gaussianized.h5" \
    --simba-train-data "$simba/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$simba/derived/hong2021_v12/simba_cv16_23_train_gaussianized.h5" \
    --tng-validation-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
    --tng-validation-cache "$tng/derived/hong2021_v12/tng100_validation_gaussianized.h5" \
    --simba-validation-data "$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5" \
    --simba-validation-cache "$simba/derived/hong2021_v12/simba_cv24_26_validation_gaussianized.h5" \
    --out "$out" --steps 10000 --validation-every 500 --workers 1 --device cuda
