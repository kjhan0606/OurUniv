#!/usr/bin/env bash
#SBATCH --job-name=CF4_Q1_alias_kernel
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
cd /home/kjhan/BACKUP/CF4
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts" JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_q1_alias_kernel_check.py
