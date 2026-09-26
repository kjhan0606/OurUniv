#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_z0_gf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
snapshot="$run/output_00003"
gf=/home/kjhan/BACKUP/GalaxyFinder
decision=/home/kjhan/BACKUP/CF4/config/cf4_lg_s40349_zoom_l19_z0_decision_v1.json
istep=3
nsplit=512
nranks="${SLURM_NTASKS:?}"
step=$(printf '%05d' "$istep")

test -x "$gf/NewDD/newdd.exe"
test -x "$gf/opFoF/opfof.exe"
test -s "$snapshot/info_${step}.txt"
test -s "$decision"
grep -q '"decision": "TRACE_ONLY_ZOOM_L19_Z0_FORWARD_PASS"' "$decision"
source_ncpu=$(awk '$1=="ncpu" {print $3; exit}' "$snapshot/info_${step}.txt")
(( source_ncpu % nranks == 0 ))

cd "$run"
test ! -e "FoF_Data/NewDD.${step}"
test ! -e "FoF_Data/FoF.${step}"
mkdir -p FoF_Data FoF_Garbage
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
export NEWDD_DM_ONLY=1 NEWDD_BOXLEN_CMPCH=384
{
  printf 'job=%s host=%s run=%s snapshot=%s step=%s nsplit=%s ranks=%s source_ncpu=%s dm_only=true box_cMpc_h=384 variable_particle_mass=true\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$run" "$snapshot" "$istep" "$nsplit" "$nranks" "$source_ncpu"
  printf 'galaxyfinder_git_head='; git -C "$gf" rev-parse HEAD
  sha256sum "$gf/NewDD/newdd.exe" "$gf/opFoF/opfof.exe" \
    "$gf/NewDD/newdd.c" "$gf/NewDD/rd_part.c" \
    "$gf/opFoF/opfof.c" "$gf/opFoF/Rules.make" \
    "$snapshot/info_${step}.txt" "$decision"
} | tee "galaxyfinder_provenance_${step}.txt"

srun --mpi=pmi2 --ntasks="$nranks" "$gf/NewDD/newdd.exe" "$istep" "$nsplit" \
  2>&1 | tee "galaxyfinder_newdd_${step}.log"
newdd_dir="FoF_Data/NewDD.${step}"
test -d "$newdd_dir"
dm_files=$(find "$newdd_dir" -maxdepth 1 -type f -name "SN.${step}.DM.*.dat" | wc -l)
info_files=$(find "$newdd_dir" -maxdepth 1 -type f -name "SN.${step}.*.info" | wc -l)
(( dm_files == nsplit ))
(( info_files == nsplit ))

srun --mpi=pmi2 --ntasks="$nranks" "$gf/opFoF/opfof.exe" "$istep" "$nsplit" \
  2>&1 | tee "galaxyfinder_opfof_${step}.log"
catalog="FoF_Data/FoF.${step}/FoF_halo_cat.${step}"
members="FoF_Data/FoF.${step}/FoF_member_particle.${step}"
test -s "$catalog"
test -s "$members"
sha256sum "$catalog" "$members" > "${catalog}.sha256"
newdd_bytes=$(du -sb "$newdd_dir" | awk '{print $1}')
catalog_bytes=$(stat -c '%s' "$catalog")
member_bytes=$(stat -c '%s' "$members")
printf 'GALAXYFINDER_RAMSES_OPFOF_PASS job=%s run=%s step=%s nsplit=%s dm_files=%s info_files=%s newdd_bytes=%s catalog_bytes=%s member_bytes=%s\n' \
  "$SLURM_JOB_ID" "$run" "$istep" "$nsplit" "$dm_files" "$info_files" \
  "$newdd_bytes" "$catalog_bytes" "$member_bytes" | tee "galaxyfinder_result_${step}.txt"
