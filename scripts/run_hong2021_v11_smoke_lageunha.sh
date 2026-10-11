#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
out=$tng/training/tng100_simba_v11_recentered_smoke
cd "$repo"
export PYTHONPATH=$repo/src
python src/hong2021_residual_v11_recentered.py train \
    --initialize "$tng/training/tng100_simba_v8_observable_context/validation_checkpoints/step_010000.pt" \
    --tng-train-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
    --tng-train-cache "$tng/derived/hong2021_v11/tng100_train_corrected_fullband.h5" \
    --simba-train-data "$simba/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5" \
    --simba-train-cache "$simba/derived/hong2021_v11/simba_cv16_23_train_corrected_fullband.h5" \
    --tng-validation-data "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
    --tng-validation-cache "$tng/derived/hong2021_v11/tng100_validation_corrected_fullband.h5" \
    --simba-validation-data "$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5" \
    --simba-validation-cache "$simba/derived/hong2021_v11/simba_cv24_26_validation_corrected_fullband.h5" \
    --out "$out" --steps 4 --validation-every 2 --batch 2 \
    --validation-batch 2 --smoke-limit 2 --workers 0 --device cuda
python src/hong2021_residual_v11_recentered.py sample \
    --data "$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5" \
    --cache "$simba/derived/hong2021_v11/simba_cv24_26_validation_corrected_fullband.h5" \
    --checkpoint "$out/validation_checkpoints/step_000004.pt" \
    --out "$out/smoke_ensemble.h5" --indices 0 --ensemble 2 \
    --sampling-steps 2 --device cuda
