#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_z0_ngf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

source /etc/profile.d/lmod.sh
module purge
module load intel/tbb/2022.3 intel/umf/1.0.2 \
  intel/compiler-rt/2025.3.0 intel/compiler/2025.3.0 \
  intel/dev-utilities/2025.3.0 intel/mpi/2021.17 gnu13/13.2.0

run=/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268
source_tree=/gpfs/kjhan/CF4/external/GalaxyFinder_newgal_586a62e
binary="$source_tree/NewGalFinder/gfind_nmeg8000.exe"
plan=/home/kjhan/BACKUP/CF4/CF4_S40349_NEWGALFINDER_PLAN.md
istep=3
step=$(printf '%05d' "$istep")
fof_dir="$run/FoF_Data/FoF.${step}"
catalog="$fof_dir/FoF_halo_cat.${step}"
members="$fof_dir/FoF_member_particle.${step}"
checksum_file="${catalog}.sha256"
list="$fof_dir/GALCATALOG.LIST.${step}"
data="$fof_dir/GALFIND.DATA.${step}"
background="$fof_dir/background_ptl.${step}"
result="$run/newgalfinder_result_${step}.txt"
binary_sha=948a2ae6049933ade00f72168dbc05c10eeaba776e7022429a921b074c00d0ad
source_commit=586a62e09d5a06cc94b88748cd01e8215f569710

test -x "$binary"
test -s "$catalog"
test -s "$members"
test -s "$checksum_file"
test -s "$plan"
test "$(git -C "$source_tree" rev-parse HEAD)" = "$source_commit"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
test ! -e "$list"
test ! -e "$data"
test ! -e "$background"
test ! -e "$result"

cd "$run"
sha256sum -c "$checksum_file"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OMP_STACKSIZE=256M
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm
export I_MPI_HYDRA_BOOTSTRAP=fork

{
  printf 'job=%s host=%s run=%s step=%s mpi_ranks=%s omp_threads=%s memory=150G\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$run" "$istep" "$SLURM_NTASKS" \
    "$SLURM_CPUS_PER_TASK"
  printf 'source_commit=%s\nbinary_sha256=%s\nworker_allocator_mb=8000\n' \
    "$source_commit" "$binary_sha"
  cat "$checksum_file"
  sha256sum "$plan"
} | tee "newgalfinder_provenance_${step}.txt"

srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" "$binary" "$istep" \
  2>&1 | tee "newgalfinder_${step}.log"

test -s "$list"
test -s "$data"
test -e "$background"
if grep -Eiq 'segmentation fault|out of memory|cannot allocate|fatal|error initializing' \
    "newgalfinder_${step}.log"; then
  printf 'NewGalFinder fatal pattern found in log\n' >&2
  exit 1
fi

list_bytes=$(stat -c '%s' "$list")
data_bytes=$(stat -c '%s' "$data")
background_bytes=$(stat -c '%s' "$background")
catalog_lines=$(wc -l < "$list")
sha256sum "$list" "$data" "$background" > "${data}.sha256"
printf 'NEWGALFINDER_PASS job=%s run=%s step=%s catalog_lines=%s list_bytes=%s data_bytes=%s background_bytes=%s\n' \
  "$SLURM_JOB_ID" "$run" "$istep" "$catalog_lines" "$list_bytes" \
  "$data_bytes" "$background_bytes" | tee "$result"
