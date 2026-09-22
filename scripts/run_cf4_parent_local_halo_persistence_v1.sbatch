#!/usr/bin/env bash
#SBATCH --job-name=cf4_parent_lvenv
#SBATCH --partition=a10,a40,a100_pcie,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

cd /home/kjhan/BACKUP/CF4
export PYTHONPATH=/home/kjhan/BACKUP/CF4/scripts:/home/kjhan/BACKUP/CF4/src
/home/kjhan/miniconda3/envs/circle/bin/python3.11 -m pytest -q \
  tests/test_cf4_parent_local_halo_persistence.py
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_parent_local_halo_persistence.py \
  --parent-catalogue-glob '/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_p2_l9_v1/halos_p3292_s*.npz' \
  --conditioned-comparison /gpfs/kjhan/CF4/diagnostics/cf4_lg_environment_screen_comparison_v1.json \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_parent_b3292_local_halo_persistence_v1.json
