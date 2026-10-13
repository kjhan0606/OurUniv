#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
model=$tng/derived/hong2021_v14/model
out=$tng/training/tng100_simba_swift_v15_e3_relative_noise_tail_quarter_edm
cd "$repo"
export PYTHONPATH=$repo/src

python src/hong2021_v15_edm.py \
  --experiment e3 --registry config/hong2021_v15_development_program.json \
  --repo "$repo" \
  --tng-train-data "$tng/derived/hong2021_v14/cic_data/tng100_train.h5" \
  --tng-train-cache "$model/tng_train_standardized.h5" \
  --tng-validation-data "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
  --tng-validation-cache "$model/tng_validation_standardized.h5" \
  --simba-train-data "$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5" \
  --simba-train-cache "$model/simba_train_standardized.h5" \
  --simba-validation-data "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
  --simba-validation-cache "$model/simba_validation_standardized.h5" \
  --swift-train-data "$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5" \
  --swift-train-cache "$model/swift_eagle_train_standardized.h5" \
  --swift-validation-data "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
  --swift-validation-cache "$model/swift_eagle_validation_standardized.h5" \
  --out "$out" --workers 1 --device cuda
