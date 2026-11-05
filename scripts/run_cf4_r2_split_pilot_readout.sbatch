#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_split_readout
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
module purge
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_PILOT_DIR:?}" "${CF4_R2_OUT_DIR:?}"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- scripts/cf4_r2_split_pilot_readout.py scripts/run_cf4_r2_split_pilot_readout.sbatch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_split_pilot_readout.py
