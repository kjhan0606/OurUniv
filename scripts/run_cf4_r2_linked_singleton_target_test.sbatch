#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_linked_target_test
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
: "${CF4_EXPECTED_COMMIT:?submit with current implementation commit}"
EXPECTED_CODE_COMMIT=$CF4_EXPECTED_COMMIT
git diff --quiet "$EXPECTED_CODE_COMMIT" -- \
  src/cf4_r2_count_exposure.py \
  src/cf4_r2_linked_singleton_jax.py \
  src/cf4_r2_linked_singleton_target.py \
  tests/test_cf4_r2_count_exposure.py \
  tests/test_cf4_r2_linked_singleton_jax.py \
  tests/test_cf4_r2_linked_singleton_target.py \
  tests/test_cf4_r2_marked_tracer_jax.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_count_exposure.py
"$PY" tests/test_cf4_r2_linked_singleton_jax.py
"$PY" tests/test_cf4_r2_linked_singleton_target.py
exec "$PY" tests/test_cf4_r2_marked_tracer_jax.py
