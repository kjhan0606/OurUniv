#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_rayresume
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
module purge
cd /home/kjhan/BACKUP/CF4
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$CF4_EXPECTED_COMMIT" ]]; then
  printf 'source commit mismatch: expected %s, found %s\n' \
    "$CF4_EXPECTED_COMMIT" "$actual_commit" >&2
  exit 2
fi
if [[ ! -f "$CF4_R2_OUT_DIR/data_free_result.json" \
      || ! -f "$CF4_R2_OUT_DIR/selection_nside1024.h5" \
      || -e "$CF4_R2_OUT_DIR/training_impact.json" ]]; then
  printf 'refusing incomplete or already-consumed checkpoint: %s\n' \
    "$CF4_R2_OUT_DIR" >&2
  exit 3
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/kjhan/miniconda3/bin/python3.13 -m pytest -q \
  tests/test_cf4_r2_selection_ray_integral.py \
  tests/test_cf4_r2_shell_geometry_census.py \
  tests/test_cf4_r2_ray_selection_training_impact.py
export CF4_R2_RESUME_DATA_FREE=1
exec /home/kjhan/miniconda3/bin/python3.13 \
  scripts/cf4_r2_ray_selection_training_impact.py
