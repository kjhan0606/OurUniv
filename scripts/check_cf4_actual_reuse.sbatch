#!/usr/bin/env bash
#SBATCH --job-name=cf4_A_reuse
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2400M
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:scripts:tests
export CF4_POSTERIOR_PLAN=config/cf4_actual_data_longer_v3.json
/home/kjhan/miniconda3/envs/circle/bin/python3.11 -m unittest test_cf4_actual_data_preview
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_actual_data_preview.py check-reuse
