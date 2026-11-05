#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_H100_graph_v6
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
: "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_r2_graph_closed_split.py
