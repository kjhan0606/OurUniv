#!/usr/bin/env bash
# Rebuild the lagRamses reader/preflight binary with the checked-in
# maxiter_fine namelist support.  The existing build tree is never modified.
#SBATCH --job-name=cf4_build_maxiter
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
src=/home/kjhan/BACKUP/lagRamses-de-nonstd
build=/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax_maxiter30
if [[ ! -e "$build" ]]; then cp -a "$src/build_lb_minimax" "$build"; fi
cd "$build"
make -j8 USE_FFTW=1 ramses
cp -f ramses_final3d "$build/ramses_lb_minimax3d_maxiter30"
chmod 755 "$build/ramses_lb_minimax3d_maxiter30"
sha256sum "$build/ramses_lb_minimax3d_maxiter30"
