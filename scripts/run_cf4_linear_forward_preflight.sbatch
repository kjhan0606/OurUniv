#!/usr/bin/env bash
#SBATCH --job-name=cf4_pm_preflight
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2400M
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/kjhan/BACKUP/CF4/src
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_linear_forward_preflight.py --output /gpfs/kjhan/CF4/kf_design/pm_linear_forward_preflight_v1/result.json
