#!/usr/bin/env bash
#SBATCH --job-name=cf4_z2_operator
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1200M
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Estimated peak 1000 MiB with 20 percent requested headroom. No GPU request.
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_z2_operator_necessity.py --task "$SLURM_ARRAY_TASK_ID"
