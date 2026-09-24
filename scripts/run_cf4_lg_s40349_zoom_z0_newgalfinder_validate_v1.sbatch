#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_ngf_val
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268
catalog="$run/FoF_Data/FoF.00003/GALCATALOG.LIST.00003"
finder_result="$run/newgalfinder_result_00003.txt"
output=/gpfs/kjhan/CF4/diagnostics/cf4_lg_s40349_zoom_l19_z0_newgalfinder_validation_v1.json
config="$root/config/p2_lg_targets_v11_parent_b3292_beta025_l9.json"
reference=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_v1.json

test -s "$finder_result"
grep -q '^NEWGALFINDER_PASS ' "$finder_result"
test -s "$catalog"
test -s "$config"
test -s "$reference"
test ! -e "$output"

cd "$root"
python scripts/cf4_newgalfinder_zoom_validation.py \
  --catalog "$catalog" \
  --config "$config" \
  --parent-reference "$reference" \
  --output "$output"
test -s "$output"
sha256sum "$output" > "${output}.sha256"
