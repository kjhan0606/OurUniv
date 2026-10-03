#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_l12_read
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
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_zoom_l12_reader_v1.nml"
ic=/gpfs/kjhan/CF4/zoom/cf4_lg_b3292_s40349_traceonly_l12_v2
ic_report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_zoom_ic_l12_v2.json
run_root=/gpfs/kjhan/CF4/ramses/s40349_zoom_l12_reader_v1
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -s "$source_namelist"
test -s "$ic_report"
for level in 009 010 011 012; do
  for field in ic_deltab ic_velcx ic_velcy ic_velcz ic_refmap; do
    test -s "$ic/level_${level}/$field"
  done
done
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$ic_report" "$ic" <<'PY'
import json
import struct
import sys
from pathlib import Path

report = json.load(open(sys.argv[1]))
root = Path(sys.argv[2])
assert report["decision"] == "TRACE_ONLY_ZOOM_IC_L12_GENERATED_PREFLIGHT_PENDING"
assert report["parent_promoted"] is False
assert report["m33_resolved"] is False
assert report["grafic_header_payload_bytes"] == 48
assert report["grafic_header_omega_b"] == 0.0
for level in range(9, 13):
    for name in ("ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz", "ic_refmap"):
        with (root / f"level_{level:03d}" / name).open("rb") as stream:
            size = struct.unpack("<i", stream.read(4))[0]
            payload = stream.read(size)
            end = struct.unpack("<i", stream.read(4))[0]
        if size != 48 or end != 48 or struct.unpack("<f", payload[44:48])[0] != 0.0:
            raise SystemExit(f"extended omega_b header mismatch: L{level} {name}")
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
cd "$run"
available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L12-reader parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s ic=%s noutput=1 aout=1.1 expected_outputs=0 measured_uniform_L9_peak_GiB=36.4 estimated_peak_GiB=48 requested_mem_GiB=64 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$ic" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$ic/zoom_ic_manifest.json" "$ic_report" \
  | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 900s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log
grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
grep -q 'FFT direct solve DONE' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge|outside.*initial|severe' ramses.log
if find "$run" -maxdepth 1 -type d -name 'output_*' -print -quit | grep -q .; then
  echo 'reader gate unexpectedly wrote a full snapshot' >&2
  exit 1
fi
printf 'TRACE_ONLY_ZOOM_L12_READER_PASS job=%s run=%s outputs=0\n' \
  "$SLURM_JOB_ID" "$run" | tee result.txt
