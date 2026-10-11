#!/usr/bin/env bash
#SBATCH --job-name=cf4_lgenv_compare
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
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_compare_lg_environment_screens.py \
  --strict-result /gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_conditioned_environment_v3/result.json \
  --pair-result /gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_pair_diagnostic_v3/result.json \
  --pair-catalogue-dir /gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_pair_diagnostic_v3 \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_environment_screen_comparison_v1.json
