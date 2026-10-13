#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_anglocal
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$CF4_EXPECTED_COMMIT" ]]; then
  printf 'source commit mismatch: expected %s, found %s\n' "$CF4_EXPECTED_COMMIT" "$actual_commit" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
/home/kjhan/miniconda3/bin/python3.13 -m pytest -q \
  tests/test_cf4_r2_v6_pixel_angular_local.py
/home/kjhan/miniconda3/bin/python3.13 \
  scripts/cf4_r2_v6_pixel_angular_setup.py
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_r2_v6_pixel_angular_local.py
