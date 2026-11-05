#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_n512
#SBATCH --partition=h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_parent_candidate_n256_stability.py \
  --candidate /gpfs/kjhan/CF4/recon/linear_cr/parent_ensemble_conditioning_refine_v1/cf4_parent_ensemble_conditioned_b3292_beta0p25.npz \
  --p1-config config/p1_targets_v2_observer.json \
  --N 512 --cluster-only --seeds 2001 2002 2003 2004 \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_parent_candidate_b3292_beta0p25_n512_stability_v1.json
