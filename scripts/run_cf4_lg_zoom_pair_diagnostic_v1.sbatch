#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_zoom_pair
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
output=/gpfs/kjhan/CF4/ramses/cf4_lg_zoom_z0_production_v1/job_388664/output_00002
tags=/gpfs/kjhan/CF4/hop/cf4_lg_zoom_z0_v1/job_390703/hop_tags_for_catalog.tag
out=/gpfs/kjhan/CF4/diagnostics/cf4_lg_zoom_pair_v1.json
test -r "$output/info_00002.txt"; test -r "$tags"
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
python /home/kjhan/BACKUP/CF4/src/cf4_lg_zoom_pair_diagnostic.py --output "$output" --hop-tags "$tags" --out "$out" | tee /gpfs/kjhan/CF4/logs/cf4_lg_zoom_pair_result_${SLURM_JOB_ID}.txt
printf 'LG_ZOOM_PAIR_DIAGNOSTIC_PASS job=%s out=%s\n' "$SLURM_JOB_ID" "$out"
