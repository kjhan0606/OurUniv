#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_beta_refine
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_refine_parent_ensemble_beta.py \
  --coarse-manifest /gpfs/kjhan/CF4/recon/linear_cr/parent_ensemble_conditioning_pilot_v2/manifest.json \
  --base /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3/cf4_linear_cr_v3_bgc_parent_extension_v3_s3292.npz \
  --outdir /gpfs/kjhan/CF4/recon/linear_cr/parent_ensemble_conditioning_refine_v1
