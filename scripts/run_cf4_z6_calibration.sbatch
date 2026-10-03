#!/usr/bin/env bash
#SBATCH --job-name=cf4_z6_calibrate
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2400M
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# Expected 2000 MiB plus 20%; no GPU or new PM evolution.
python_bin=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$python_bin" -m unittest discover -s tests -p test_cf4_z0_physical_field.py -v
"$python_bin" -m unittest discover -s tests -p test_cf4_pm_calibrated_z0.py -v
"$python_bin" -m unittest discover -s tests -p test_cf4_chunked_hmc.py -v
exec "$python_bin" scripts/cf4_z6_native_physics.py
