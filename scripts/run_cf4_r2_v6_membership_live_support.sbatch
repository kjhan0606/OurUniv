#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_v6_support
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$CF4_EXPECTED_COMMIT" -- scripts/cf4_r2_v6_membership_live_support.py scripts/run_cf4_r2_v6_membership_live_support.sbatch
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_v6_membership_live_support.py
