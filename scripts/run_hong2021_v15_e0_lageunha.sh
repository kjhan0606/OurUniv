#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
model=$tng/derived/hong2021_v14/model
training=$tng/training/tng100_simba_swift_v14_multiscale_edm
evaluation=$tng/evaluation/tng100_simba_swift_v14_multiscale_edm
out=$tng/evaluation/tng100_simba_swift_v15/e0_v14_diagnostic.json
cd "$repo"
export PYTHONPATH=$repo/src

simba_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_simba_dev_representative16_v1.json"))["indices"])))')
swift_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_swift_eagle_dev_representative16_v1.json"))["indices"])))')

python src/hong2021_v15_diagnostics.py \
  --registry config/hong2021_v15_development_program.json \
  --v14-decision "$evaluation/development_decision.json" \
  --training "$training" --evaluation "$evaluation" \
  --tng-data "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
  --tng-cache "$model/tng_validation_standardized.h5" \
  --tng-indices 49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62 \
  --simba-data "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
  --simba-cache "$model/simba_validation_standardized.h5" \
  --simba-indices "$simba_indices" \
  --swift-data "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
  --swift-cache "$model/swift_eagle_validation_standardized.h5" \
  --swift-indices "$swift_indices" \
  --steps 2000 5000 10000 --sigmas 0.5 2 8 32 \
  --seeds 25173 25174 25175 --batch 2 --device cuda --out "$out"
