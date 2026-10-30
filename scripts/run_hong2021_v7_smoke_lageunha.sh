#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
derived=$simba/derived/hong2021_v1
deterministic=$tng/training/tng100_v4_split00_l0_groupnorm_std_cosine/minimum_validation_loss.pt
parent=$tng/training/tng100_v6_edm_laplacian_sigma2/minimum_validation.pt
tng_train=$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5
tng_validation=$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5
tng_train_cache=$tng/derived/hong2021_v6/tng100_train_laplacian_sigma2.h5
tng_validation_cache=$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5
simba_train=$derived/simba_cv16_23_train_all_observers.h5
simba_validation=$derived/simba_cv24_26_validation_all_observers.h5
simba_train_mean=$derived/simba_cv16_23_train_deterministic_k2_4.h5
simba_validation_mean=$derived/simba_cv24_26_validation_deterministic_k2_4.h5
simba_train_cache=$derived/simba_cv16_23_train_laplacian_sigma2.h5
simba_validation_cache=$derived/simba_cv24_26_validation_laplacian_sigma2.h5
out=$tng/training/tng100_simba_v7_multidomain_edm_smoke
sample=$simba/evaluation/hong2021_v7_smoke_sample.h5

cd "$repo"
export PYTHONPATH=$repo/src

if [[ ! -s "$simba_train_mean" ]]; then
    python src/hong2021_residual_diffusion.py prepare \
        --data "$simba_train" --checkpoint "$deterministic" \
        --out "$simba_train_mean" --batch 6 --workers 1 --device cuda
fi
if [[ ! -s "$simba_validation_mean" ]]; then
    python src/hong2021_residual_diffusion.py prepare \
        --data "$simba_validation" --checkpoint "$deterministic" \
        --out "$simba_validation_mean" --batch 6 --workers 1 --device cuda
fi
if [[ ! -s "$simba_train_cache" ]]; then
    python src/hong2021_residual_v6.py prepare \
        --data "$simba_train" --mean-cache "$simba_train_mean" \
        --out "$simba_train_cache" --sigma-cells 2 --chunk 8
fi
if [[ ! -s "$simba_validation_cache" ]]; then
    python src/hong2021_residual_v6.py prepare \
        --data "$simba_validation" --mean-cache "$simba_validation_mean" \
        --out "$simba_validation_cache" --sigma-cells 2 --chunk 8
fi

python src/hong2021_residual_v7_multidomain.py \
    --initialize "$parent" \
    --tng-train-data "$tng_train" --tng-train-cache "$tng_train_cache" \
    --simba-train-data "$simba_train" --simba-train-cache "$simba_train_cache" \
    --tng-validation-data "$tng_validation" \
    --tng-validation-cache "$tng_validation_cache" \
    --simba-validation-data "$simba_validation" \
    --simba-validation-cache "$simba_validation_cache" \
    --out "$out" --steps 20 --batch 6 --validation-batch 6 \
    --workers 1 --validation-every 10 --smoke-limit 12 --device cuda

python src/hong2021_residual_v6.py sample \
    --data "$simba_validation" --cache "$simba_validation_cache" \
    --checkpoint "$out/minimum_validation.pt" --out "$sample" \
    --indices 0 --ensemble 2 --sampling-steps 4 --seed 3777 --device cuda
