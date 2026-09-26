#!/usr/bin/env bash
#SBATCH --job-name=cf4_lgenv3_select
#SBATCH --partition=a100_pcie,h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
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
  scripts/cf4_select_and_reproduce_lg_environment_candidate.py \
  --search-result /gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_conditioned_environment_v3/result.json \
  --base-config config/cf4_lg_highk_conditioning_pilot_b3292_v3_environment.json \
  --selected-config /gpfs/kjhan/CF4/recon/linear_cr/cf4_lg_highk_selected_environment_v3.json \
  --output /gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_selected_environment_v3
