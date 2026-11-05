#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_LF_skyhold
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/home/kjhan/miniconda3/bin/python -m pytest -q tests/test_cf4_r2_lf_conditional.py
exec /home/kjhan/miniconda3/bin/python scripts/cf4_r2_lf_conditional_check.py
