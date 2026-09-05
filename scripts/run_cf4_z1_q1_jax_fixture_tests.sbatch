#!/usr/bin/env bash
#SBATCH --job-name=cf4_z1_q1_jax
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=2400M
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this runner with sbatch}"
cd /home/kjhan/BACKUP/CF4
# Estimated process upper bound 2000 MiB plus 20 percent headroom.
git rev-parse HEAD
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  tests/test_cf4_q1_cell_integrated_convolution_jax.py
