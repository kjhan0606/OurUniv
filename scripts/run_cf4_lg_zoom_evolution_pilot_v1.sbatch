#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_zoom_evol
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax_maxiter30/ramses_lb_minimax3d_maxiter30
namelist=/home/kjhan/BACKUP/CF4/config/ramses_cf4_lg_zoom_evolution_pilot_v1.nml
ic=/gpfs/kjhan/CF4/zoom/cf4_lg_zoom_pilot_v3
root=/gpfs/kjhan/CF4/ramses/cf4_lg_zoom_evolution_pilot_v1
run_dir="$root/job_${SLURM_JOB_ID}"
test -x "$binary"; test -r "$namelist"
for level in 008 009 010 011 012; do
  for field in ic_deltab ic_velcx ic_velcy ic_velcz; do test -r "$ic/level_${level}/$field"; done
done
mkdir -p "$root"; mkdir "$run_dir"; cd "$run_dir"
printf 'job=%s host=%s binary=%s namelist=%s ic=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$binary" "$namelist" "$ic" | tee provenance.txt
sha256sum "$binary" "$namelist" "$ic/zoom_ic_manifest.json" | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 13000s mpirun -np 4 "$binary" "$namelist" 2>&1 | tee ramses.log
grep -Eq 'Run completed|Total elapsed time' ramses.log
if grep -q 'Some grid are outside initial conditions sub-volume' ramses.log; then echo 'ZOOM_EVOL_FAIL_BOUNDARY' >&2; exit 4; fi
if grep -q 'Fine multigrid Poisson failed to converge' ramses.log; then echo 'ZOOM_EVOL_FAIL_POISSON' >&2; exit 5; fi
if grep -Eiq 'nan|negative density|negative pressure' ramses.log; then echo 'ZOOM_EVOL_FAIL_STATE' >&2; exit 6; fi
test -r output_00002/info_00002.txt
awk '$1=="aexp" {if (($3+0) < 0.05) exit 1; found=1} END {exit(found ? 0 : 1)}' output_00002/info_00002.txt
printf 'ZOOM_EVOL_PASS job=%s run_dir=%s\n' "$SLURM_JOB_ID" "$run_dir" | tee result.txt
