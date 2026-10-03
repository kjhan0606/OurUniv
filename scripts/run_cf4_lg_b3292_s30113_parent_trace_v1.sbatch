#!/usr/bin/env bash
#SBATCH --job-name=cf4_s30113_trace
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s30113_parent_z0_v1/job_394245

python "$root/scripts/cf4_trace_ramses_member_ids.py" \
  --snapshot "$run/output_00001" \
  --members /gpfs/kjhan/CF4/masks/cf4_lg_b3292_s30113_parent_members_v1.npz \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s30113_lagrangian_trace_v1.json \
  --positions-output /gpfs/kjhan/CF4/masks/cf4_lg_b3292_s30113_lagrangian_positions_v1.npz \
  --box 384
