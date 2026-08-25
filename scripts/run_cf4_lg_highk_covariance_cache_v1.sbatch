#!/bin/bash
#SBATCH --job-name=cf4_highk_covcache
#SBATCH --partition=h200,h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=19G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_covariance_cache_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_covariance_cache_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly program="$repo/config/cf4_lg_highk_streaming_forward_program_v1.json"
readonly expected_program_sha=34a8e8976be7b5b8690568b620898910d534913ca8326cb5f86120f63cfcd8b1
readonly expected_source_sha=6da87bb7ef73ada54269424c82e949d16059a87eb209bc79235144681979040d

[[ "$(sha256sum "$program" | awk '{print $1}')" == "$expected_program_sha" ]]
[[ "$(sha256sum "$repo/src/cf4_lg_highk_covariance_cache.py" | awk '{print $1}')" == "$expected_source_sha" ]]

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec /home/kjhan/miniconda3/bin/python -m cf4_lg_highk_covariance_cache \
    --program "$program"
