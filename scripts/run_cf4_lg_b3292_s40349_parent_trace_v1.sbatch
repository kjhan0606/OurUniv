#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_trace
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=${CF4_S40349_PRODUCTION_RUN:?set production run directory}
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  "$root/scripts/cf4_trace_ramses_member_ids.py" \
  --snapshot "$run/output_00001" \
  --members /gpfs/kjhan/CF4/masks/cf4_lg_b3292_i248_s40349_survival_members_v1.npz \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_lagrangian_trace_v1.json \
  --positions-output /gpfs/kjhan/CF4/masks/cf4_lg_b3292_s40349_lagrangian_positions_v1.npz \
  --box 384.0
