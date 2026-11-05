#!/usr/bin/env bash
#SBATCH --job-name=cf4_virgo_hr_pilot
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec /home/kjhan/miniconda3/bin/python3.13 \
  scripts/cf4_virgo_hr_constraint_pilot.py \
  --field /gpfs/kjhan/CF4/kf_design/low_high_splice_repair_bundle_v1/repaired_field.npz \
  --output /gpfs/kjhan/CF4/kf_design/virgo_hr_constraint_pilot_v1/result.json \
  --radius 5.0 --targets 1.0 1.5 1.686 2.0 --joint-preserve
