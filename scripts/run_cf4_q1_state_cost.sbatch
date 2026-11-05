#!/usr/bin/env bash
#SBATCH --job-name=CF4_Q1_state_cost
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:$PWD/tests" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_q1_state_cost.py
