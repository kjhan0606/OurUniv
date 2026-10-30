#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_point_mark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
/home/kjhan/miniconda3/bin/python3.13 -m unittest -q tests.test_cf4_r2_point_process
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_r2_point_mark_manifest.py
