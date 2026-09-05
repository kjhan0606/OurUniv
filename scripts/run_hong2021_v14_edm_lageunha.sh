#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
root=$tng/derived/hong2021_v14/model
status=$root/preparation_status.json
out=$tng/training/tng100_simba_swift_v14_multiscale_edm
cd "$repo"
export PYTHONPATH=$repo/src

while true; do
    if [[ -s $status ]]; then
        state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$status")
        if [[ $state == complete ]]; then break; fi
        if [[ $state == failed* ]]; then
            printf 'Residual preparation failed: %s\n' "$state" >&2
            exit 1
        fi
    fi
    sleep 30
done

python src/hong2021_v14_edm.py train \
    --tng-train-data "$tng/derived/hong2021_v14/cic_data/tng100_train.h5" \
    --tng-train-cache "$root/tng_train_standardized.h5" \
    --tng-validation-data "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
    --tng-validation-cache "$root/tng_validation_standardized.h5" \
    --simba-train-data "$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$root/simba_train_standardized.h5" \
    --simba-validation-data "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
    --simba-validation-cache "$root/simba_validation_standardized.h5" \
    --swift-train-data "$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5" \
    --swift-train-cache "$root/swift_eagle_train_standardized.h5" \
    --swift-validation-data "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
    --swift-validation-cache "$root/swift_eagle_validation_standardized.h5" \
    --out "$out" --steps 10000 --candidate-steps 2000,5000,10000 \
    --batch 6 --validation-batch 6 --workers 1 --device cuda
