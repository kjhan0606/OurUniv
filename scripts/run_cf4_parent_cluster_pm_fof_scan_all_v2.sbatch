#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_pmfof_all
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_parent_cluster_pm_fof_scan.py \
  --manifest /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3/manifest_v3_bgc_parent_extension_v3.json \
  --physical-scan /gpfs/kjhan/CF4/diagnostics/cf4_parent_physical_cluster_gate_scan_v1.json \
  --p1-config config/p1_targets_v2_observer.json \
  --all-members \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_parent_cluster_pm_fof_scan_all_v2.json
