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
export CF4_SOURCE_COMMIT="$(git rev-parse HEAD)"
staged_archive=${CF4_SDSS_MOCK_STAGED_ARCHIVE:-}
if [ -n "$staged_archive" ]; then
    archive="$staged_archive"
    workdir="${archive%/mocks.tar.gz}"
    [[ "$archive" == "$workdir/mocks.tar.gz" ]]
    [[ "$workdir" =~ ^/gpfs/kjhan/CF4/external/sdss_pv_6824749/\.cf4_sdss_mocks_[0-9]+\.[A-Za-z0-9]{6}$ ]]
    [[ -f "$archive" ]]
else
    workdir=$(mktemp -d "/gpfs/kjhan/CF4/external/sdss_pv_6824749/.cf4_sdss_mocks_${SLURM_JOB_ID}.XXXXXX")
    chmod 700 "$workdir"
    archive="$workdir/mocks.tar.gz"
fi
success=0
cleanup() {
    if [ "$success" -eq 1 ]; then
        rm -- "$archive"
        rmdir -- "$workdir"
    fi
}
trap cleanup EXIT
if [ -z "$staged_archive" ]; then
    curl --silent --show-error --fail --location --retry 5 --retry-all-errors --retry-delay 10 \
        --continue-at - --output "$archive" \
        https://zenodo.org/records/6824749/files/mocks.tar.gz
fi
/home/kjhan/miniconda3/bin/python3.13 scripts/cf4_r2_sdss_mock_stream_summary.py \
    --archive "$archive" \
    --output-dir "/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_stream_law_${SLURM_JOB_ID}_20261005"
success=1
