#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_selection_coordinate
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/bin/python3.13
"$PY" -m unittest discover -s tests -p test_cf4_r2_observed_magnitude_transfer.py -q
exec "$PY" scripts/cf4_r2_selection_coordinate_control.py
