#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_l19_a010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_zoom_l19_early_a010_v2.nml"
source_run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_early_v1/job_1108467
source_output="$source_run/output_00001"
source_decision="$root/config/cf4_lg_s40349_zoom_l19_early_a005_decision_v1.json"
run_root=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_early_a010_v2
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -s "$source_namelist"
test -s "$source_output/info_00001.txt"
test -s "$source_decision"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$source_decision" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1]))
assert decision["decision"] == "INCONCLUSIVE_EPOCH_TOO_EARLY_NO_REFINEMENT_ABOVE_L12"
assert decision["parent_promoted"] is False
assert decision["m33_resolved"] is False
assert decision["checks"]["ramses_completed"] is True
assert decision["checks"]["maximum_refined_level_seen"] == 12
assert decision["checks"]["full_snapshots"] == 1
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
ln -s "$source_output" "$run/output_00001"
cd "$run"

grep -Eq '^nrestart=1$' ramses.nml
grep -Eq '^noutput=1$' ramses.nml
grep -Eq '^aout=0\.10$' ramses.nml
grep -Eq '^match_aout=\.true\.$' ramses.nml
grep -Eq '^tout=1\.d100$' ramses.nml
grep -Eq '^foutput=1000000000$' ramses.nml
grep -Eq '^fbackup=1000000000$' ramses.nml

available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L19-early-restart parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s restart=%s target_a=0.10 noutput=1 expected_new_outputs=1 estimated_new_output_GiB=24 measured_a005_peak_GiB=63.76 estimated_peak_GiB=80 requested_mem_GiB=96 ntasks=32 cpus_per_task=2 time_limit=02:00:00 nparttot=190000000 ngridtot=48000000 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$source_output" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$source_output/info_00001.txt" \
  "$source_decision" | tee -a provenance.txt

export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 6900s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log

grep -q 'Restarting at' ramses.log
grep -q 'Run completed' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge|outside.*initial|severe|Restart failed' ramses.log

test -L "$run/output_00001"
test -d "$run/output_00002"
mapfile -t new_outputs < <(find "$run" -mindepth 1 -maxdepth 1 -type d -name 'output_*' | sort)
if [[ ${#new_outputs[@]} -ne 1 || ${new_outputs[0]} != "$run/output_00002" ]]; then
  printf 'expected only one new physical output_00002 directory\n' >&2
  exit 1
fi
info="$run/output_00002/info_00002.txt"
test -s "$info"
a_final=$(awk '$1=="aexp" {print $3; exit}' "$info")
awk -v a="$a_final" 'BEGIN {exit !(a >= 0.10 && a < 0.14)}'
max_level=$(awk '/Level[[:space:]]+[0-9]+ has/ {for (i=1;i<=NF;i++) if ($i=="Level") {x=$(i+1)+0; if (x>m) m=x}} END {print m+0}' ramses.log)
if (( max_level <= 12 )); then
  printf 'a=0.10 extension still did not refine above L12: max_level=%d\n' "$max_level" >&2
  exit 1
fi
output_bytes=$(find "$run/output_00002" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f",s}')
printf 'TRACE_ONLY_ZOOM_L19_EARLY_A010_PASS job=%s run=%s a_final=%s max_level=%s new_outputs=1 output_bytes=%s\n' \
  "$SLURM_JOB_ID" "$run" "$a_final" "$max_level" "$output_bytes" | tee result.txt
