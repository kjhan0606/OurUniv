#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_surv_cause
#SBATCH --partition=a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- scripts/cf4_r2_survival_cause_attribution.py scripts/cf4_r2_survival_holdout_screen.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_survival_cause_attribution.py
