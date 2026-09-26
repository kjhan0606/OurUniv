#!/usr/bin/env bash
#SBATCH --job-name=cf4_z6_posterior
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9831M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
# Permit unrelated status commits during the calibration dependency.
git diff --quiet "$EXPECTED_COMMIT" -- src/cf4_z0_physical_field.py src/cf4_pm_calibrated_z0.py src/cf4_chunked_hmc.py scripts/cf4_z6_native_physics.py scripts/cf4_z5_bias_posterior.py config/cf4_z6_native_physics_plan_v1.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
export CF4_POSTERIOR_PLAN=config/cf4_z6_native_physics_plan_v1.json
unset JAX_PLATFORMS
# Expected host peak 8192 MiB, request ceil(1.2*8192)=9831 MiB.
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_z5_bias_posterior.py
