#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
transform=$tng/derived/hong2021_v12/balanced_train_gaussianization.json
cd "$repo"
export PYTHONPATH=$repo/src

python src/hong2021_residual_v12_gaussianized.py fit-transform \
    --tng-train-cache "$tng/derived/hong2021_v11/tng100_train_corrected_fullband.h5" \
    --simba-train-cache "$simba/derived/hong2021_v11/simba_cv16_23_train_corrected_fullband.h5" \
    --out "$transform" --histogram-bins 131072 --knots 8193 --z-limit 5

prepare_one() {
    local source=$1 output=$2
    python src/hong2021_residual_v12_gaussianized.py prepare \
        --v11-cache "$source" --transform "$transform" --out "$output"
}
prepare_one \
    "$tng/derived/hong2021_v11/tng100_train_corrected_fullband.h5" \
    "$tng/derived/hong2021_v12/tng100_train_gaussianized.h5"
prepare_one \
    "$simba/derived/hong2021_v11/simba_cv16_23_train_corrected_fullband.h5" \
    "$simba/derived/hong2021_v12/simba_cv16_23_train_gaussianized.h5"
prepare_one \
    "$tng/derived/hong2021_v11/tng100_validation_corrected_fullband.h5" \
    "$tng/derived/hong2021_v12/tng100_validation_gaussianized.h5"
prepare_one \
    "$simba/derived/hong2021_v11/simba_cv24_26_validation_corrected_fullband.h5" \
    "$simba/derived/hong2021_v12/simba_cv24_26_validation_gaussianized.h5"
