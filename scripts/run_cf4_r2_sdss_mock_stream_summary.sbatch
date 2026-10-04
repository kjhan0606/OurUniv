#!/usr/bin/env bash
# CPU-only streaming audit; it does not allocate a GPU.
#SBATCH --job-name=cf4_R2_sdss_mocks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a10
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_stream_summary_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_stream_summary_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src:/home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
workdir=$(mktemp -d "/scratch/cf4_kjhan_sdss_mocks_${SLURM_JOB_ID}.XXXXXX")
chmod 700 "$workdir"
archive="$workdir/mocks.tar.gz"
success=0
cleanup() {
    if [ "$success" -eq 1 ]; then
        rm -- "$archive"
        rmdir -- "$workdir"
    fi
}
trap cleanup EXIT
curl --fail --location --retry 5 --retry-all-errors --retry-delay 10 \
    --continue-at - --output "$archive" \
    https://zenodo.org/records/6824749/files/mocks.tar.gz
/home/kjhan/miniconda3/bin/python3.13 scripts/cf4_r2_sdss_mock_stream_summary.py \
    --archive "$archive" \
    --output-dir "/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_stream_law_${SLURM_JOB_ID}_20261005"
success=1
