#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_grad28
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}" "${CF4_GPU_MODE:?}"
case "$CF4_GPU_MODE" in
  h200) FREE_NAME=CF4_H200_FREE_TYPED_GPUS ;;
  h100) FREE_NAME=CF4_H100_FREE_TYPED_GPUS ;;
  a100) FREE_NAME=CF4_A100_FREE_TYPED_GPUS ;;
  *) echo "unsupported GPU mode: $CF4_GPU_MODE" >&2; exit 1 ;;
esac
FREE="${!FREE_NAME:?}"
[[ "$FREE" =~ ^[0-9]+$ ]]
if (( FREE == 0 )); then
  [[ "${CF4_GPU_QUEUE_OCCUPIED:-}" == 1 ]]
fi
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  src/cf4_r2_resolution_target.py \
  scripts/cf4_r2_conditional_map.py \
  scripts/cf4_r2_conditional_optimizer_diagnostic.py \
  scripts/run_cf4_r2_conditional_full_gradient28.sbatch \
  tests/test_cf4_r2_conditional_optimizer_diagnostic.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_FULL_GRADIENT28=1
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_conditional_optimizer_diagnostic.py -q
exec "$PY" scripts/cf4_r2_conditional_optimizer_diagnostic.py
