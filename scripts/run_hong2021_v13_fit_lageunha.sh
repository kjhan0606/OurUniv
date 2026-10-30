#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
out=$tng/derived/hong2021_v13/observable_dc_ridge.json

cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_dc_correction.py fit \
    --tng-train-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
    --tng-train-cache "$tng/derived/hong2021_v11/tng100_train_corrected_fullband.h5" \
    --simba-train-data "$simba/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$simba/derived/hong2021_v11/simba_cv16_23_train_corrected_fullband.h5" \
    --out "$out" --folds 5 \
    --regularizations 0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1 \
    --seed 13021
