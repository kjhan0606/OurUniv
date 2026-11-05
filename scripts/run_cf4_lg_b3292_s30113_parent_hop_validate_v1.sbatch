#!/usr/bin/env bash
#SBATCH --job-name=cf4_s30113_hopval
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s30113_parent_z0_v1/job_394245

python "$root/scripts/cf4_hop_parent_environment.py" \
  --hop "$run/HOP_00002/grp00002.pos" \
  --galaxyfinder /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s30113_parent_galaxyfinder_v1.json \
  --p1 "$root/config/p1_targets_v2_observer.json" \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s30113_parent_hop_v1.json
