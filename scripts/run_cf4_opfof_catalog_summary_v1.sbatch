#!/usr/bin/env bash
#SBATCH --job-name=cf4_opfof_summary
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
python /home/kjhan/BACKUP/CF4/src/cf4_opfof_catalog_summary.py \
  --catalog /gpfs/kjhan/CF4/diagnostics/cf4_lg_zoom_z0_opfof_v1/opfof_catalog.npz \
  --out /gpfs/kjhan/CF4/diagnostics/cf4_lg_zoom_z0_opfof_summary_v1.json
