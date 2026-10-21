#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CAMELS/SIMBA/L25n256
data=$root/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5
cache=$root/derived/hong2021_v1/simba_cv16_23_train_laplacian_sigma2.h5
checkpoint=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_v8_observable_context/validation_checkpoints/step_010000.pt
out=$root/evaluation/hong2021_v8_simba_train_diagnostic
cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$out"
python src/hong2021_select_observable_representatives.py \
  --data "$data" --cache "$cache" --feature-fit-checkpoint "$checkpoint" \
  --count 16 --out "$out/representative16.json"
indices=$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["indices"])))' "$out/representative16.json")
python src/hong2021_residual_v8_context.py sample \
  --data "$data" --cache "$cache" --checkpoint "$checkpoint" \
  --out "$out/ensemble16_steps40.h5" --indices "$indices" \
  --ensemble 16 --sampling-steps 40 --seed 12777 --device cuda
python src/hong2021_residual_evaluate.py \
  --candidate "edm=$out/ensemble16_steps40.h5" \
  --out "$out/ensemble_evaluation" --voxel-mpc-h 0.3125
