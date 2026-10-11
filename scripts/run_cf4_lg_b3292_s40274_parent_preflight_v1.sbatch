#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40274_rpre
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
readonly binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
readonly namelist=/home/kjhan/BACKUP/CF4/config/ramses_cf4_lg_b3292_s40274_parent_preflight_v1.nml
readonly ic=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_selected_pair_v3/grafic_l9
readonly ic_link=/gpfs/kjhan/CF4/ramses/ic/s40274_parent
readonly run_root=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s40274_parent_preflight_v1
readonly run_dir="$run_root/job_${SLURM_JOB_ID}"
test -x "$binary"
test -r "$namelist"
for field in ic_deltab ic_velcx ic_velcy ic_velcz; do test -r "$ic/$field"; done
mkdir -p "$(dirname "$ic_link")"
if [[ -L "$ic_link" ]]; then
  [[ "$(readlink -f "$ic_link")" == "$(readlink -f "$ic")" ]]
elif [[ -e "$ic_link" ]]; then
  echo "existing short IC path is not the expected symlink: $ic_link" >&2
  exit 3
else
  ln -s "$ic" "$ic_link"
fi
mkdir -p "$run_root"
mkdir "$run_dir"
cd "$run_dir"
printf 'job=%s host=%s binary=%s namelist=%s ic=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$binary" "$namelist" "$ic" | tee provenance.txt
sha256sum "$binary" "$namelist" "$ic/ic_deltab" "$ic/ic_velcx" \
  "$ic/ic_velcy" "$ic/ic_velcz" | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=15s 1500s \
  mpirun -np 32 "$binary" "$namelist" 2>&1 | tee ramses.log
grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
grep -q 'FFT direct solve DONE' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge' ramses.log
! find "$run_dir" -maxdepth 1 -type d -name 'output_*' | grep -q .
printf 'PARENT_PREFLIGHT_PASS job=%s run_dir=%s\n' "$SLURM_JOB_ID" "$run_dir" | tee result.txt
