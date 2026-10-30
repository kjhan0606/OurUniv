#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_joint
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}" "${CF4_GPU_MODE:?}"
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  src/cf4_affine_objective.py src/cf4_r2_resolution_target.py \
  scripts/cf4_r2_conditional_map.py scripts/cf4_r2_joint_warm_start.py \
  scripts/run_cf4_r2_joint_warm_start.sbatch \
  tests/test_cf4_affine_objective.py tests/test_cf4_r2_conditional_map_budget.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_affine_objective.py -q
"$PY" tests/test_cf4_r2_conditional_map_budget.py -q
"$PY" tests/test_cf4_r2_linked_point_conditioning_radius.py -q
"$PY" tests/test_cf4_r2_count_exposure.py -q
exec "$PY" scripts/cf4_r2_joint_warm_start.py
