#!/usr/bin/env bash
# RAMSES -> GalaxyFinder NewDD -> RAMSES-aware variable-mass opFoF.
# This deliberately does not use src/opfof/opfof (the old GOTPM bridge).
#SBATCH --job-name=cf4_galaxyfinder_ramses
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

run=/gpfs/kjhan/CF4/ramses/cf4_lg_zoom_z0_production_v1/job_388664
gf=/gpfs/kjhan/CF4/external/GalaxyFinder
istep=2
nsplit=64

cd "$run"
mkdir -p FoF_Data FoF_Garbage

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_FABRICS=shm
export I_MPI_HYDRA_BOOTSTRAP=fork
export NEWDD_DM_ONLY=1
export NEWDD_BOXLEN_CMPCH=384

mpirun -np 8 "$gf/NewDD/newdd.exe" "$istep" "$nsplit"
mpirun -np 8 "$gf/opFoF/opfof.exe" "$istep" "$nsplit"

test -s "FoF_Data/FoF.$(printf '%05d' "$istep")/FoF_halo_cat.$(printf '%05d' "$istep")"
printf 'GALAXYFINDER_RAMSES_OPFOF_PASS job=%s run=%s step=%s nsplit=%s\n' "${SLURM_JOB_ID:?}" "$run" "$istep" "$nsplit"
