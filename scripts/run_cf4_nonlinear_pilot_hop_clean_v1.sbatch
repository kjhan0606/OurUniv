#!/usr/bin/env bash
#SBATCH --job-name=cf4_nonlinear_hop_clean
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
hop=/gpfs/kjhan/CF4/hop/clean_rebuild_v1/job_388420/hop
regroup=/gpfs/kjhan/CF4/hop/clean_rebuild_v1/job_388420/regroup
source=/gpfs/kjhan/CF4/ramses/cf4_production_ic_nonlinear_pilot_v1/job_388412/output_00004
root=/gpfs/kjhan/CF4/hop/cf4_nonlinear_pilot_v1_clean
work="$root/job_${SLURM_JOB_ID}"
test -x "$hop"; test -x "$regroup"; test -r "$source/part_00004.out00001"
mkdir -p "$root"; mkdir "$work"; cd "$work"
sha256sum "$hop" "$regroup" > provenance.sha256
printf 'job=%s host=%s source=%s hop=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$source" "$hop" | tee provenance.txt
stdbuf -oL -eL "$hop" -in "$source/part_00004.out" -p 1. -nd 64 -nh 64 -nm 4 -o hop > hop.log 2>&1
stdbuf -oL -eL "$regroup" -root hop -douter 80. -dsaddle 200. -dpeak 240. -f77 -o grp > regroup.log 2>&1
test -s grp.tag
printf 'HOP_CLEAN_PRELIMINARY_PASS job=%s source=%s work=%s\n' "$SLURM_JOB_ID" "$source" "$work" | tee result.txt
