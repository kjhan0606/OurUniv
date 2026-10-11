#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_ares_bridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_r2_ares_catalogue_bridge.py
