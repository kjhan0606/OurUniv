#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_rate_grad128
#SBATCH --partition=a100_pcie,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- scripts/cf4_r2_rate_nuisance_n128_screen.py src/cf4_r2_rate_marginal.py config/cf4_r2_rate_nuisance_v1.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_rate_nuisance_n128_screen.py
