#!/usr/bin/env bash
#SBATCH --job-name=cf4_b3292_lgpair3
#SBATCH --partition=a100_pcie,h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_lg_highk_conditioning_stream.py \
  --config config/cf4_lg_highk_conditioning_b3292_v3_pair_diagnostic.json
