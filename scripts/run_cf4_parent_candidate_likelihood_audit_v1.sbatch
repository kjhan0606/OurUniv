#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_like
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_parent_candidate_likelihood_audit.py \
  --parent-manifest /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3/manifest_v3_bgc_parent_extension_v3.json \
  --base /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3/cf4_linear_cr_v3_bgc_parent_extension_v3_s3292.npz \
  --candidate /gpfs/kjhan/CF4/recon/linear_cr/parent_ensemble_conditioning_refine_v1/cf4_parent_ensemble_conditioned_b3292_beta0p25.npz \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_parent_candidate_b3292_beta0p25_likelihood_audit_v1.json
