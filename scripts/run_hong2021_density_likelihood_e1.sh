#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
derived=$simba/derived/hong2021_v1
out=$tng/evaluation/hong2021_density_likelihood_e1

cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_transfer_likelihood.py \
    --source tng \
        "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
        "$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5" sample \
    --source simba \
        "$derived/simba_cv16_23_train_all_observers.h5" \
        "$derived/simba_cv16_23_train_deterministic_k2_4.h5" realization \
    --source simba \
        "$derived/simba_cv24_26_validation_all_observers.h5" \
        "$derived/simba_cv24_26_validation_deterministic_k2_4.h5" realization \
    --out "$out" --bootstrap 5000 --seed 482021
