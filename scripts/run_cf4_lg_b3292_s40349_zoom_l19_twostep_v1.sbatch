#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_l19_2step
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_zoom_l19_twostep_v1.nml"
ic=/gpfs/kjhan/CF4/zoom/cf4_lg_b3292_s40349_traceonly_l12_v2
ic_report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_zoom_ic_l12_v2.json
reader_run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l12_reader_v1/job_1108348
reader_decision="$root/config/cf4_lg_s40349_zoom_l12_reader_decision_v1.json"
run_root=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_twostep_v1
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -s "$source_namelist"
test -s "$ic_report"
test -s "$reader_run/result.txt"
test -s "$reader_decision"
grep -q 'TRACE_ONLY_ZOOM_L12_READER_PASS' "$reader_run/result.txt"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$reader_decision" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1]))
assert decision["decision"] == "TRACE_ONLY_ZOOM_L12_READER_PASS"
assert decision["parent_promoted"] is False
assert decision["m33_resolved"] is False
assert decision["checks"]["unexpected_full_snapshots"] == 0
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
cd "$run"
available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L19-two-step parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s ic=%s nstepmax=2 noutput=1 aout=1.1 expected_outputs=0 measured_reader_peak_GiB=42.0 estimated_peak_GiB=53 requested_mem_GiB=64 nparttot=190000000 ngridtot=24000000 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$ic" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$ic/zoom_ic_manifest.json" "$ic_report" \
  "$reader_decision" | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 900s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log
grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
grep -q 'FFT direct solve DONE' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge|outside.*initial|severe' ramses.log
if find "$run" -maxdepth 1 -type d -name 'output_*' -print -quit | grep -q .; then
  echo 'two-step gate unexpectedly wrote a full snapshot' >&2
  exit 1
fi
printf 'TRACE_ONLY_ZOOM_L19_TWOSTEP_PASS job=%s run=%s outputs=0\n' \
  "$SLURM_JOB_ID" "$run" | tee result.txt
