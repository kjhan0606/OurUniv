#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_traceonly
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
source_namelist="$root/config/ramses_cf4_lg_b3292_s40349_trace_only_v1.nml"
ic=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/i248_s40349/grafic_l9
ic_link=/gpfs/kjhan/CF4/ramses/ic/sv1/i248_s40349
members=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_i248_s40349_survival_members_v1.npz
galaxyfinder=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_v1.json
hop=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_hop_nh16_regroup_v2.json
run_root=/gpfs/kjhan/CF4/ramses/s40349_traceonly
run="$run_root/job_${SLURM_JOB_ID}"

test -x "$binary"
test -r "$source_namelist"
test -s "$members"
test -s "$galaxyfinder"
test -s "$hop"
for field in ic_deltab ic_velcx ic_velcy ic_velcz; do test -r "$ic/$field"; done
[[ -L "$ic_link" ]]
[[ "$(readlink -f "$ic_link")" == "$(readlink -f "$ic")" ]]
/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$galaxyfinder" "$hop" <<'PY'
import json
import sys

galaxyfinder = json.load(open(sys.argv[1]))
hop = json.load(open(sys.argv[2]))
if galaxyfinder["decision"] != "PARENT_RAMSES_HALO_PASS":
    raise SystemExit("GalaxyFinder seed-40349 pair evidence is not passing")
if galaxyfinder["lg"].get("selected") is None or galaxyfinder["gates"].get("standard_lg_isolation") is not True:
    raise SystemExit("GalaxyFinder selected-pair or isolation evidence is missing")
raw = hop["lg"]["raw_peak_crossmatch"]
merged = hop["environment"]["pair_crossmatch"]
if not (raw["distinct_raw_peaks"] and raw["both_within_1_mpc_h"]):
    raise SystemExit("HOP raw peaks do not support trace-only candidate status")
if not (merged["mode"] == "merged_hop_group_mass_consistent" and hop["decision"] == "HOP_PARENT_STRUCTURE_FAIL"):
    raise SystemExit("HOP regroup qualification changed; refuse trace-only launch")
PY

mkdir -p "$run_root"
mkdir "$run"
cp "$source_namelist" "$run/ramses.nml"
cd "$run"
available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-zoom-candidate parent_promoted=false m33_resolved=false binary=%s effective_namelist=%s ic=%s noutput=1 aout=0.02 measured_GiB_each=18 expected_GiB_total=18 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$run/ramses.nml" "$ic" "$available" \
  | tee provenance.txt
sha256sum "$binary" "$run/ramses.nml" "$ic/ic_deltab" "$ic/ic_velcx" \
  "$ic/ic_velcy" "$ic/ic_velcz" "$members" "$galaxyfinder" "$hop" \
  | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 1500s \
  mpirun -np "$SLURM_NTASKS" "$binary" "$run/ramses.nml" 2>&1 | tee ramses.log
grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
grep -q 'FFT direct solve DONE' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge' ramses.log
mapfile -t outputs < <(find "$run" -maxdepth 1 -type d -name 'output_*' | sort -V)
[[ "${#outputs[@]}" -eq 1 ]]
initial="${outputs[0]}"
number="${initial##*_}"
aexp=$(awk '$1 == "aexp" {print $3; exit}' "$initial/info_${number}.txt")
awk -v a="$aexp" 'BEGIN {exit !(a >= 0.019999999 && a <= 0.020000001)}'

/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  "$root/scripts/cf4_trace_ramses_member_ids.py" \
  --snapshot "$initial" --members "$members" \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_lagrangian_trace_v2.json \
  --positions-output /gpfs/kjhan/CF4/masks/cf4_lg_b3292_s40349_lagrangian_positions_v2.npz \
  --box 384.0 --selection-status trace-only-zoom-candidate \
  --evidence "$galaxyfinder" --evidence "$hop"
printf 'TRACE_ONLY_LAGRANGIAN_HANDOFF_PASS job=%s initial=%s aexp=%s trace=%s\n' \
  "$SLURM_JOB_ID" "$initial" "$aexp" \
  /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_lagrangian_trace_v2.json \
  | tee result.txt
