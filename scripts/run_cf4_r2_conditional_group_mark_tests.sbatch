#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_conditional_mark_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kjhan/miniconda3/bin/python3.13 -m pytest -q tests/test_cf4_r2_conditional_group_mark.py
