#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_GL2_chain
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
: "${CF4_R2_CHAIN_LABEL:?set CF4_R2_CHAIN_LABEL to a or b}"
: "${CF4_R2_OUT_DIR:?set a unique CF4_R2_OUT_DIR}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_EXACT_CHAIN_STEPS="${CF4_R2_EXACT_CHAIN_STEPS:-8}"
export CF4_R2_CHAIN_INTEGRATION_STEPS="${CF4_R2_CHAIN_INTEGRATION_STEPS:-1}"
export CF4_R2_FUNDAMENTAL_MASS="${CF4_R2_FUNDAMENTAL_MASS:-6000}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" -m unittest discover -s tests -p test_cf4_r2_prior_split_hmc.py -q
exec "$PY" scripts/cf4_r2_force_pair_comparison.py
