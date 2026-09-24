#!/usr/bin/env bash
#SBATCH --job-name=cf4_tng_match
#SBATCH --partition=a10,a40,a100_pcie,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
module purge
module load gnu14
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet -- scripts/cf4_tng_fetch_match.py scripts/cf4_tng_hydro_dark_match.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/kjhan/miniconda3/bin/python3.13
"$PY" -m pytest -q tests/test_cf4_tng_hydro_dark_match.py
OUT=/gpfs/kjhan/CF4/calibration/tng_hydro_dark_v1
if [ ! -f "$OUT/subhalo_matching_to_dark.hdf5" ]; then
  "$PY" scripts/cf4_tng_fetch_match.py \
    --key-file /home/kjhan/.config/OurUniv/tng_api_key \
    --output "$OUT/subhalo_matching_to_dark.hdf5"
fi
"$PY" scripts/cf4_tng_hydro_dark_match.py \
  --match "$OUT/subhalo_matching_to_dark.hdf5" \
  --hydro-dir /scratch/kjhan/IllustrisTNG/TNG100-1/output/groups_099 \
  --output "$OUT/summary.json"
