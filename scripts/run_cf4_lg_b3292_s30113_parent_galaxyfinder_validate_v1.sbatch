#!/usr/bin/env bash
#SBATCH --job-name=cf4_s30113_gfval
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s30113_parent_z0_v1/job_394245
fof="$run/FoF_Data/FoF.00002"

python "$root/scripts/cf4_galaxyfinder_parent_validation.py" \
  --catalog "$fof/FoF_halo_cat.00002" \
  --members "$fof/FoF_member_particle.00002" \
  --p1 "$root/config/p1_targets_v2_observer.json" \
  --p2 "$root/config/p2_lg_targets_v11_parent_b3292_beta025_l9.json" \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s30113_parent_galaxyfinder_v1.json \
  --member-output /gpfs/kjhan/CF4/masks/cf4_lg_b3292_s30113_parent_members_v1.npz
