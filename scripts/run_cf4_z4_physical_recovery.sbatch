#!/usr/bin/env bash
#SBATCH --job-name=cf4_z4_physical
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4800M
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# CPU only: estimated 4000 MiB per task plus 20 percent headroom.
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    /home/kjhan/miniconda3/envs/circle/bin/python3.11 -m unittest discover -s tests -p test_cf4_z0_physical_field.py -v
fi
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_z4_physical_recovery.py
