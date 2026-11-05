#!/usr/bin/env bash
set -euo pipefail

cd /home/kjhan/BACKUP/CF4

while tmux has-session -t hong2021_v6_compare 2>/dev/null; do
  sleep 30
done

python_bin=/home/kjhan/miniconda3/bin/python
root=/gpfs/kjhan/IllustrisTNG/TNG100-1
data=${root}/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5
cache=${root}/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5
training=${root}/training
evaluation=${root}/evaluation/tng100_v6_edm_flow
indices=49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62

mkdir -p "${evaluation}"

for method in edm flow; do
  checkpoint=${training}/tng100_v6_${method}_laplacian_sigma2/minimum_validation.pt
  test -f "${checkpoint}"
  PYTHONPATH=src "${python_bin}" src/hong2021_residual_v6.py sample \
    --data "${data}" \
    --cache "${cache}" \
    --checkpoint "${checkpoint}" \
    --out "${evaluation}/${method}_minimum_validation_representative16_ensemble16.h5" \
    --indices "${indices}" \
    --ensemble 16 \
    --sampling-steps 40 \
    --seed 777 \
    --device cuda
done

PYTHONPATH=src "${python_bin}" src/hong2021_residual_evaluate.py \
  --candidate "edm=${evaluation}/edm_minimum_validation_representative16_ensemble16.h5" \
  --candidate "flow=${evaluation}/flow_minimum_validation_representative16_ensemble16.h5" \
  --out "${evaluation}/ensemble_evaluation" \
  --voxel-mpc-h 0.3125
