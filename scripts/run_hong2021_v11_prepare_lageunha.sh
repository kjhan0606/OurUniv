#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
correction=$tng/training/tng100_simba_v10_twocomponent/validation_checkpoints/step_005000.pt
cd "$repo"
export PYTHONPATH=$repo/src

prepare_one() {
    local data=$1 original=$2 output=$3
    python src/hong2021_residual_v11_recentered.py prepare \
        --data "$data" --original-cache "$original" \
        --correction-checkpoint "$correction" --out "$output" \
        --batch 4 --workers 1 --device cuda
}

prepare_one \
    "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
    "$tng/derived/hong2021_v6/tng100_train_laplacian_sigma2.h5" \
    "$tng/derived/hong2021_v11/tng100_train_corrected_fullband.h5"
prepare_one \
    "$simba/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5" \
    "$simba/derived/hong2021_v1/simba_cv16_23_train_laplacian_sigma2.h5" \
    "$simba/derived/hong2021_v11/simba_cv16_23_train_corrected_fullband.h5"
prepare_one \
    "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
    "$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5" \
    "$tng/derived/hong2021_v11/tng100_validation_corrected_fullband.h5"
prepare_one \
    "$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5" \
    "$simba/derived/hong2021_v1/simba_cv24_26_validation_laplacian_sigma2.h5" \
    "$simba/derived/hong2021_v11/simba_cv24_26_validation_corrected_fullband.h5"
