#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_zoom_hop
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
source=/gpfs/kjhan/CF4/ramses/cf4_lg_zoom_z0_production_v1/job_388664/output_00002
hop=/gpfs/kjhan/CF4/hop/clean_rebuild_v1/job_388420/hop
regroup=/gpfs/kjhan/CF4/hop/clean_rebuild_v1/job_388420/regroup
root=/gpfs/kjhan/CF4/hop/cf4_lg_zoom_z0_v1
work="$root/job_${SLURM_JOB_ID}"
test -x "$hop"; test -x "$regroup"; test -r "$source/part_00002.out00001"
mkdir -p "$root"; mkdir "$work"; cd "$work"
printf 'job=%s host=%s source=%s hop=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$source" "$hop" | tee provenance.txt
sha256sum "$hop" "$regroup" "$source/namelist.txt" | tee -a provenance.txt
stdbuf -oL -eL "$hop" -in "$source/part_00002.out" -p 1. -nd 64 -nh 64 -nm 4 -o hop > hop.log 2>&1
stdbuf -oL -eL "$regroup" -root hop -douter 80. -dsaddle 200. -dpeak 240. -f77 -o grp > regroup.log 2>&1
test -s hop.hop
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
python /home/kjhan/BACKUP/CF4/scripts/cf4_z0_hop_mass_diagnostic.py \
  --output "$source" --hop "$work" \
  --out /gpfs/kjhan/CF4/diagnostics/cf4_lg_zoom_z0_hop_mass_v1.json > diagnostic.log 2>&1
printf 'LG_ZOOM_HOP_DIAGNOSTIC_PASS job=%s work=%s\n' "$SLURM_JOB_ID" "$work" | tee result.txt
