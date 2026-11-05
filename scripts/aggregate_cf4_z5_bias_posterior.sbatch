#!/usr/bin/env bash
#SBATCH --job-name=cf4_z5_finish
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1200M
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
# Allow later status-only commits; refuse changed executable/model/plan sources.
git diff --quiet "$EXPECTED_COMMIT" -- scripts/cf4_z5_bias_posterior.py scripts/cf4_z4_physical_recovery.py src/cf4_chunked_hmc.py src/cf4_z0_physical_field.py config/cf4_z5_bias_posterior_plan_v1.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
# CPU-only, 1000 MiB import/aggregation allowance + 20%.
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_z5_bias_posterior.py aggregate
