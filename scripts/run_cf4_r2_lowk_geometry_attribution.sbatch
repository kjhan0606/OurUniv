#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_lowk_geometry
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_OUT_DIR=/gpfs/kjhan/CF4/z0_density/r2_n256_lowk_geometry_attribution_20261002_v1
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" -m unittest discover -s tests -p test_cf4_r2_lowk_geometry_attribution.py -q
exec "$PY" scripts/cf4_r2_lowk_geometry_attribution.py
