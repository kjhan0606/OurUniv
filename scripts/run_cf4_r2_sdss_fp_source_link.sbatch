#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_FP_source
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_fp_distance.py
exec "$PY" scripts/cf4_r2_sdss_fp_source_link.py
