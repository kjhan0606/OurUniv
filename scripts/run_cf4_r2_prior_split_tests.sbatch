#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_split_mechanics
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
module purge
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
git diff --exit-code "$CF4_EXPECTED_COMMIT" -- scripts/cf4_r2_prior_split_hmc.py scripts/cf4_r2_split_sampling_run.py tests/test_cf4_r2_prior_split_hmc.py scripts/run_cf4_r2_prior_split_tests.sbatch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_prior_split_hmc.py
