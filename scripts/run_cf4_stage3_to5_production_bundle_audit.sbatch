#!/usr/bin/env bash
#SBATCH --job-name=cf4_s3_s5_audit
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2400M
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_stage3_to5_production_bundle_audit.py --output /gpfs/kjhan/CF4/kf_design/stage3_to5_production_bundle_audit_v1/result.json
