#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40274_hop
#SBATCH --partition=a10,a40,a100_pcie,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
ulimit -c 0

run=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s40274_parent_z0_v1/job_396128
hoproot=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
out=HOP_00002
cd "$run"
mkdir -p "$out"
sha256sum "$hoproot/hop" "$hoproot/regroup" "$hoproot/poshalo" > "$out/binary_SHA256SUMS"
"$hoproot/hop" -in output_00002/part_00002.out -p 1.0 -o "$out/hop00002"
"$hoproot/regroup" -root "$out/hop00002" -douter 80.0 -dsaddle 200.0 \
  -dpeak 240.0 -f77 -o "$out/grp00002"
"$hoproot/poshalo" -inp output_00002 -pre "$out/grp00002" \
  -xmi 0.0 -xma 1.0 -ymi 0.0 -yma 1.0 -zmi 0.0 -zma 1.0 -per 1.0
test -s "$out/grp00002.pos"
sha256sum "$out"/* > "$out/output_SHA256SUMS"
printf 'RAMSES_HOP_PASS job=%s run=%s\n' "${SLURM_JOB_ID:?}" "$run"
