#!/usr/bin/env bash
#SBATCH --job-name=cf4_hop_rebuild_smoke
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
src=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
build_root=/gpfs/kjhan/CF4/hop/clean_rebuild_v1
build="$build_root/job_${SLURM_JOB_ID}"
source=/gpfs/kjhan/CF4/ramses/cf4_production_ic_nonlinear_pilot_v1/job_388412/output_00004
mkdir -p "$build_root"; mkdir "$build"
for f in Makefile hop.c hop_input.c kd.c kd.h smooth.c smooth.h regroup.c slice.c slice.h; do cp "$src/$f" "$build/$f"; done
cd "$build"
make -j1 CFLAGS='-O -mcmodel=medium -DHUGE=1.0e30' hop regroup > build.log 2>&1
test -x hop; test -x regroup
sha256sum hop regroup > rebuilt.sha256
mkdir work; cd work
stdbuf -oL -eL ../hop -in "$source/part_00004.out" -p 1. -nd 1 -nh 2 -nm 0 -densityonly -o hop > hop.log 2>&1
grep -q 'Actual number of DM particles = 16777216' hop.log
printf 'HOP_CLEAN_REBUILD_FORMAT_SMOKE_PASS job=%s build=%s\n' "$SLURM_JOB_ID" "$build" | tee result.txt
