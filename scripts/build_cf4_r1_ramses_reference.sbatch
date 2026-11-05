#!/usr/bin/env bash
#SBATCH --job-name=cf4_R1_ref_build
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- scripts config src tests
source_repo=/home/kjhan/BACKUP/lagRamses-de-nonstd
source_commit=d689044d896d9f195629891a202eaf2fd64ee0d2
build_root=/gpfs/kjhan/CF4/r1_ref_build/job_${SLURM_JOB_ID}
mkdir -p /gpfs/kjhan/CF4/r1_ref_build
mkdir "$build_root"
git -C "$source_repo" archive "$source_commit" amr hydro pm poisson aton patch utils/scripts bin/Makefile | tar -x -C "$build_root"
export OMP_NUM_THREADS=1
cd "$build_root/bin"
timeout --signal=TERM --kill-after=15s 1100s make -j1 USE_FFTW=1 EXEC=ramses_cf4_ref \
  GITBRANCH=cf4-reference GITHASH="$source_commit" GITREPO=git@github.com:kjhan0606/LagRamses.git
test -x ramses_cf4_ref3d
sha256sum ramses_cf4_ref3d
printf 'CF4_REFERENCE_BUILD_COMPLETE source=%s binary=%s/bin/ramses_cf4_ref3d\n' "$source_commit" "$build_root"
