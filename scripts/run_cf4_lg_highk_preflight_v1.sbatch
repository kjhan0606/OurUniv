#!/bin/bash
#SBATCH --job-name=cf4_highk_preflight
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_preflight_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_preflight_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/bin/python
readonly bank=/gpfs/kjhan/CF4/recon/linear_cr/lowk_parent_posterior_bridge_v1/posterior_bank.npz
readonly digest=25ec02b9ddfa736c170c6c84332845f026790b1162f4ba14ba63fc2aa34d9e76

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="$repo/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$python" -m cf4_lg_highk_schedule \
    --bank "$bank" \
    --expected-sha256 "$digest"
