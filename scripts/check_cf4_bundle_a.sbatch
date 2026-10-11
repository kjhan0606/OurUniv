#!/usr/bin/env bash
#SBATCH --job-name=cf4_A_check
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3600M
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:tests
/home/kjhan/miniconda3/envs/circle/bin/python3.11 -m unittest test_cf4_quantile_z0 test_cf4_pm_calibrated_z0 test_cf4_z0_physical_field
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_bundle_a_prior_to_data.py preflight
