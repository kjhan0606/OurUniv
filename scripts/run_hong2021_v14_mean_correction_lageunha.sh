#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
audit=$tng/evaluation/tng100_simba_swift_v14_baseline_audit
out=$tng/training/tng100_simba_swift_v14_mean_correction
cd "$repo"
export PYTHONPATH=$repo/src

python src/hong2021_v14_mean_correction.py \
    --tng-train-data "$tng/derived/hong2021_v14/cic_data/tng100_train.h5" \
    --tng-train-cache "$audit/tng_train.h5" \
    --tng-validation-data "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
    --tng-validation-cache "$audit/tng_validation.h5" \
    --simba-train-data "$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$audit/simba_train.h5" \
    --simba-validation-data "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
    --simba-validation-cache "$audit/simba_validation.h5" \
    --swift-train-data "$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5" \
    --swift-train-cache "$audit/swift_eagle_train.h5" \
    --swift-validation-data "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
    --swift-validation-cache "$audit/swift_eagle_validation.h5" \
    --out "$out" --steps 5000 --candidate-steps 1000,3000,5000 \
    --batch 6 --validation-batch 6 --workers 1 --device cuda
