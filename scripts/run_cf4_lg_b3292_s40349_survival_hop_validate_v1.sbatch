#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_hopval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/sv1_grammar/i248_s40349/job_1104302
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  "$root/scripts/cf4_hop_parent_environment.py" \
  --hop "$run/HOP_00001/grp00001.pos" \
  --galaxyfinder /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_v1.json \
  --p1 "$root/config/p1_targets_v2_observer.json" \
  --p2 "$root/config/p2_lg_targets_v11_parent_b3292_beta025_l9.json" \
  --environment-policy standard-lg-isolation \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_hop_v1.json
