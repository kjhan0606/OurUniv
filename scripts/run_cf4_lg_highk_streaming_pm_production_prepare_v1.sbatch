#!/bin/bash
#SBATCH --job-name=cf4_highk_pm_prep
#SBATCH --partition=h200,h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1_prepare-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1_prepare-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly program="$repo/config/cf4_lg_highk_streaming_forward_program_v1.json"
readonly cache=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_covariance_cache_v1/cache.npz
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1
readonly expected_program_sha=856b758457dc69f3d9e6ee19ad8b88d68da96bc3d801e6b5a9f01b87a98dd135
readonly expected_runner_sha=0494bdf3440e85906e4ded45ce9b1284fc40533b9a7c8f12aab1dac617201814
readonly expected_cache_sha=c15747dd7afdfb6d2b0013c726bdaa99b8fc1a3b3a1c75581086eebe35470b1a

[[ "$(sha256sum "$program" | awk '{print $1}')" == "$expected_program_sha" ]]
[[ "$(sha256sum "$repo/src/cf4_lg_highk_streaming_forward.py" | awk '{print $1}')" == "$expected_runner_sha" ]]
[[ "$(sha256sum "$cache" | awk '{print $1}')" == "$expected_cache_sha" ]]

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec /home/kjhan/miniconda3/bin/python \
    -m cf4_lg_highk_streaming_forward \
    --program "$program" \
    --mode production \
    --prepare-production \
    --covariance-cache "$cache" \
    --output-root "$output"
