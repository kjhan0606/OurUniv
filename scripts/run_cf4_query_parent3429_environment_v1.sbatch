#!/usr/bin/env bash
#SBATCH --job-name=cf4_p3429_env
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
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
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_query_parent3429_environment.py \
  --catalog /gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/hop_catalog_exact.npz \
  --p1-config config/p1_targets_v2_observer.json \
  --gate /gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/gate_result_v2.json \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_parent3429_existing_z0_environment_v2.json
