#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_l19_early
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
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_zoom_l19_early_v1.nml"
ic=/gpfs/kjhan/CF4/zoom/cf4_lg_b3292_s40349_traceonly_l12_v2
ic_report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_zoom_ic_l12_v2.json
startup_run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_twostep_v1/job_1108357
startup_decision="$root/config/cf4_lg_s40349_zoom_l19_twostep_decision_v1.json"
run_root=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_early_v1
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -s "$source_namelist"
test -s "$ic_report"
test -s "$startup_run/result.txt"
test -s "$startup_decision"
grep -q 'TRACE_ONLY_ZOOM_L19_TWOSTEP_PASS' "$startup_run/result.txt"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$startup_decision" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1]))
assert decision["decision"] == "TRACE_ONLY_ZOOM_L19_TWOSTEP_PASS"
assert decision["parent_promoted"] is False
assert decision["m33_resolved"] is False
assert decision["checks"]["unexpected_full_snapshots"] == 0
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
cd "$run"

# Audit the effective namelist copied into the immutable run directory.
grep -Eq '^noutput=1$' ramses.nml
grep -Eq '^aout=0\.05$' ramses.nml
grep -Eq '^match_aout=\.true\.$' ramses.nml
grep -Eq '^tout=1\.d100$' ramses.nml
grep -Eq '^foutput=1000000000$' ramses.nml
grep -Eq '^fbackup=1000000000$' ramses.nml

available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L19-early-nonlinear parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s ic=%s target_a=0.05 noutput=1 expected_outputs=1 estimated_output_GiB=32 measured_startup_peak_GiB=48.72 estimated_peak_GiB=80 requested_mem_GiB=96 ntasks=32 cpus_per_task=2 time_limit=02:00:00 nparttot=190000000 ngridtot=24000000 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$ic" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$ic/zoom_ic_manifest.json" "$ic_report" \
  "$startup_decision" | tee -a provenance.txt

export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 6900s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log

grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
grep -q 'FFT direct solve DONE' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge|outside.*initial|severe' ramses.log

mapfile -t outputs < <(find "$run" -mindepth 1 -maxdepth 1 -type d -name 'output_*' | sort)
if [[ ${#outputs[@]} -ne 1 ]]; then
  printf 'expected exactly one full snapshot, found %d\n' "${#outputs[@]}" >&2
  exit 1
fi
info=$(find "${outputs[0]}" -maxdepth 1 -type f -name 'info_*.txt' -print -quit)
test -s "$info"
a_final=$(awk '$1=="aexp" {print $3; exit}' "$info")
awk -v a="$a_final" 'BEGIN {exit !(a >= 0.05 && a < 0.08)}'
max_level=$(awk '/Level[[:space:]]+[0-9]+ has/ {for (i=1;i<=NF;i++) if ($i=="Level") {x=$(i+1)+0; if (x>m) m=x}} END {print m+0}' ramses.log)
if (( max_level <= 12 )); then
  printf 'early nonlinear gate did not refine above the loaded L12 IC: max_level=%d\n' "$max_level" >&2
  exit 1
fi
output_bytes=$(find "${outputs[0]}" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f",s}')
printf 'TRACE_ONLY_ZOOM_L19_EARLY_PASS job=%s run=%s a_final=%s max_level=%s outputs=1 output_bytes=%s\n' \
  "$SLURM_JOB_ID" "$run" "$a_final" "$max_level" "$output_bytes" | tee result.txt
