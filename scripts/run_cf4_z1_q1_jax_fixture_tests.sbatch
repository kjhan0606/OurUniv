#!/usr/bin/env bash
#SBATCH --job-name=cf4_z1_q1_jax
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/kjhan/miniconda3/envs/circle/bin/python3.11 -m pytest -q \
  tests/test_cf4_q1_cell_integrated_convolution_jax.py
