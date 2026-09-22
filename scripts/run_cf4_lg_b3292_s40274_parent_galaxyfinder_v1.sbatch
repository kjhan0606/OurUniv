#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40274_gf
#SBATCH --partition=a10,a40,a100_pcie,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

run=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_s40274_parent_z0_v1/job_396128
gf=/gpfs/kjhan/CF4/external/GalaxyFinder
istep=2
nsplit=256
nranks="${SLURM_NTASKS:?}"
step=$(printf '%05d' "$istep")

cd "$run"
mkdir -p FoF_Data FoF_Garbage
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
export NEWDD_DM_ONLY=1 NEWDD_BOXLEN_CMPCH=384
{
  printf 'run=%s\nstep=%s\nnsplit=%s\nranks=%s\n' "$run" "$istep" "$nsplit" "$nranks"
  sha256sum "$gf/NewDD/newdd.exe" "$gf/opFoF/opfof.exe"
} > "galaxyfinder_provenance_${step}.txt"
srun --mpi=pmi2 --ntasks="$nranks" "$gf/NewDD/newdd.exe" "$istep" "$nsplit"
srun --mpi=pmi2 --ntasks="$nranks" "$gf/opFoF/opfof.exe" "$istep" "$nsplit"
catalog="FoF_Data/FoF.${step}/FoF_halo_cat.${step}"
test -s "$catalog"
sha256sum "$catalog" > "${catalog}.sha256"
printf 'GALAXYFINDER_RAMSES_OPFOF_PASS job=%s run=%s step=%s nsplit=%s\n' \
  "${SLURM_JOB_ID:?}" "$run" "$istep" "$nsplit"
