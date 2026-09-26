#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_h16post
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail
ulimit -c 0

run=/gpfs/kjhan/CF4/ramses/sv1_grammar/i248_s40349/job_1104302
hoproot=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
out=HOP_00001_nh16_v2
cd "$run"
grep -q 'nSmooth = 65, nHop = 16, nMerge = 4' "$out/hop00001.gbound"
test -s "$out/hop00001.den"
test -s "$out/hop00001.hop"
test ! -e "$out/grp00001.tag"
"$hoproot/regroup" -root "$out/hop00001" -douter 80.0 -dsaddle 200.0 \
  -dpeak 240.0 -f77 -o "$out/grp00001" 2>&1 | tee "$out/regroup.log"
"$hoproot/poshalo" -inp output_00001 -pre "$out/grp00001" \
  -xmi 0.0 -xma 1.0 -ymi 0.0 -yma 1.0 -zmi 0.0 -zma 1.0 -per 1.0 \
  2>&1 | tee "$out/poshalo.log"
test -s "$out/grp00001.pos"
sha256sum "$out"/* > "$out/output_SHA256SUMS"
printf 'RAMSES_HOP_NH16_POSTPROCESS_PASS job=%s source_hop_job=1105056 output=%s\n' \
  "$SLURM_JOB_ID" "$run/$out"
