#!/usr/bin/env bash
#SBATCH --job-name=cf4_joint_cal
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2880M
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=/gpfs/kjhan/CF4/kf_design/joint_tracer_calibration_v1/result.json
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")"
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_joint_tracer_calibration_v1.py \
  --program config/cf4_joint_tracer_calibration_v1.json --output "$OUT"
