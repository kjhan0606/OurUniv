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
readonly expected_program_sha=c21e860d9e7b4ad01f3b73a6ed3304f4f2776e8fb18552b1c3e7a6674901d60c
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
