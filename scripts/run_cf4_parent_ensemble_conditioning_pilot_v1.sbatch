#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_enscond
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_parent_ensemble_conditioning_pilot.py \
  --manifest /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3/manifest_v3_bgc_parent_extension_v3.json \
  --p1-result /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v3/p1_result.json \
  --pm-fof-result /gpfs/kjhan/CF4/diagnostics/cf4_parent_cluster_pm_fof_scan_all_v2.json \
  --base-seeds 3292 \
  --outdir /gpfs/kjhan/CF4/recon/linear_cr/parent_ensemble_conditioning_pilot_v2
