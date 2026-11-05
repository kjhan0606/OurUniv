#!/usr/bin/env bash
#SBATCH --job-name=cf4_C_support
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3600M
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:scripts:tests
/home/kjhan/miniconda3/bin/python3.13 -m unittest test_cf4_bundle_c_selection_support -v
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_bundle_c_selection_support.py
