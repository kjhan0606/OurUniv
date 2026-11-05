#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_l19_z0
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

source /etc/profile.d/lmod.sh
module purge
module load intel/tbb/2022.3 intel/umf/1.0.2 \
  intel/compiler-rt/2025.3.0 intel/compiler/2025.3.0 \
  intel/dev-utilities/2025.3.0 intel/mpi/2021.17 gnu13/13.2.0

root=/home/kjhan/BACKUP/CF4
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_zoom_l19_z0_v1.nml"
source_run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_early_a010_v2/job_1108481
source_output="$source_run/output_00002"
source_decision="$root/config/cf4_lg_s40349_zoom_l19_early_a010_decision_v2.json"
run_root=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -s "$source_namelist"
test -s "$source_output/info_00002.txt"
test -s "$source_decision"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - \
  "$source_decision" "$source_output/info_00002.txt" "$SLURM_NTASKS" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1]))
assert decision["decision"] == "TRACE_ONLY_ZOOM_L19_EARLY_NONLINEAR_PASS"
assert decision["parent_promoted"] is False
assert decision["m33_resolved"] is False
assert decision["checks"]["completed"] is True
assert decision["checks"]["final_scale_factor"] == 0.1

source_ncpu = None
aexp = None
with open(sys.argv[2]) as stream:
    for line in stream:
        fields = line.split()
        if fields and fields[0] == "ncpu":
            source_ncpu = int(fields[2])
        elif fields and fields[0] == "aexp":
            aexp = float(fields[2])
assert source_ncpu == int(sys.argv[3]) == 32
assert aexp is not None and abs(aexp - 0.1) < 1.0e-10
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
ln -s "$source_output" "$run/output_00002"
cd "$run"

grep -Eq '^nrestart=2$' ramses.nml
grep -Eq '^noutput=1$' ramses.nml
grep -Eq '^aout=1\.0$' ramses.nml
grep -Eq '^match_aout=\.true\.$' ramses.nml
grep -Eq '^tout=1\.d100$' ramses.nml
grep -Eq '^foutput=1000000000$' ramses.nml
grep -Eq '^fbackup=1000000000$' ramses.nml
grep -Eq '^levelmin=9$' ramses.nml
grep -Eq '^levelmax=19$' ramses.nml
grep -Eq '^omega_b=0\.0$' ramses.nml

available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L19-z0-forward parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s restart=%s target_a=1.0 noutput=1 expected_new_outputs=1 estimated_new_output_GiB=30 measured_a010_peak_GiB=48.89 estimated_peak_GiB=80 requested_mem_GiB=96 memory_margin=20pct ntasks=32 cpus_per_task=2 mpi=intel-2021.17 compiler=intel-2025.3 gnu=13.2.0 time_limit=2-00:00:00 nparttot=190000000 ngridtot=48000000 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$source_output" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$source_output/info_00002.txt" \
  "$source_decision" | tee -a provenance.txt

export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 171000s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log

grep -q 'Restarting at' ramses.log
grep -q 'Run completed' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge|outside.*initial|severe|Restart failed' ramses.log

test -L "$run/output_00002"
test -d "$run/output_00003"
mapfile -t new_outputs < <(find "$run" -mindepth 1 -maxdepth 1 -type d -name 'output_*' | sort)
if [[ ${#new_outputs[@]} -ne 1 || ${new_outputs[0]} != "$run/output_00003" ]]; then
  printf 'expected only one new physical output_00003 directory\n' >&2
  exit 1
fi
info="$run/output_00003/info_00003.txt"
test -s "$info"
a_final=$(awk '$1=="aexp" {print $3; exit}' "$info")
awk -v a="$a_final" 'BEGIN {exit !(a >= 0.99 && a < 1.05)}'
max_level=$(awk '/Level[[:space:]]+[0-9]+ has/ {for (i=1;i<=NF;i++) if ($i=="Level") {x=$(i+1)+0; if (x>m) m=x}} END {print m+0}' ramses.log)
if (( max_level <= 12 )); then
  printf 'z=0 evolution never refined above the IC hierarchy: max_level=%d\n' "$max_level" >&2
  exit 1
fi
output_bytes=$(find "$run/output_00003" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f",s}')
printf 'TRACE_ONLY_ZOOM_L19_Z0_FORWARD_PASS job=%s run=%s a_final=%s max_level=%s new_outputs=1 output_bytes=%s parent_promoted=false m33_resolved=false\n' \
  "$SLURM_JOB_ID" "$run" "$a_final" "$max_level" "$output_bytes" | tee result.txt
