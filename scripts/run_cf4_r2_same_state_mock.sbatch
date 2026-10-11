#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_same_state
#SBATCH --partition=a40,a100_pcie,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- scripts/cf4_r2_same_state_mock.py scripts/run_cf4_r2_same_state_mock.sbatch
export PYTHONPATH="$PWD/src:$PWD/scripts" JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_same_state_mock.py
