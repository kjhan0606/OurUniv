#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_ngf_local
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gnu13/13.2.0

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268
fof="$run/FoF_Data/FoF.00003"
validation=/gpfs/kjhan/CF4/diagnostics/cf4_lg_s40349_zoom_l19_z0_newgalfinder_validation_v1.json
output=/gpfs/kjhan/CF4/diagnostics/cf4_lg_s40349_zoom_l19_z0_newgalfinder_local_audit_v1.json

test -s "$fof/GALCATALOG.LIST.00003"
test -s "$fof/GALFIND.DATA.00003"
test -s "$validation"
test ! -e "$output"

cd "$root"
/home/kjhan/miniconda3/bin/python3.13 scripts/cf4_newgalfinder_local_audit.py \
  --catalog "$fof/GALCATALOG.LIST.00003" \
  --data "$fof/GALFIND.DATA.00003" \
  --validation "$validation" \
  --config config/p2_lg_targets_v11_parent_b3292_beta025_l9.json \
  --output "$output"
test -s "$output"
sha256sum "$output" > "${output}.sha256"
