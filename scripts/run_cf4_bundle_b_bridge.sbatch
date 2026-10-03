#!/usr/bin/env bash
#SBATCH --job-name=cf4_B_bridge
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src:scripts:tests
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
unset JAX_PLATFORMS
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_bundle_b_bridge.py
