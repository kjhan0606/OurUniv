#!/usr/bin/env bash
#SBATCH --job-name=cf4_hop_format_smoke
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
hopdir=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
source=/gpfs/kjhan/CF4/ramses/cf4_production_ic_nonlinear_pilot_v1/job_388412/output_00004
root=/gpfs/kjhan/CF4/hop/cf4_nonlinear_pilot_format_smoke_v1
work="$root/job_${SLURM_JOB_ID}"
test -x "$hopdir/hop"; test -r "$source/part_00004.out00001"
mkdir -p "$root"; mkdir "$work"; cd "$work"
sha256sum "$hopdir/hop" > provenance.sha256
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$source" | tee provenance.txt
# Deliberately minimal smoothing: parser/reader smoke only, not a science catalogue.
stdbuf -oL -eL "$hopdir/hop" -in "$source/part_00004.out" -p 1. -nd 1 -nh 2 -nm 0 -densityonly -o hop > hop.log 2>&1
grep -q 'Actual number of DM particles = 16777216' hop.log
printf 'HOP_FORMAT_SMOKE_PASS job=%s work=%s\n' "$SLURM_JOB_ID" "$work" | tee result.txt
