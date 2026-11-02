#!/usr/bin/env bash
#SBATCH --job-name=cf4_r2_mala_fd
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_BASE:?}" \
  "${CF4_GPU_MODE:?}" "${CF4_SUBMIT_PARTITION:?}" "${CF4_SUBMIT_GRES:?}" \
  "${CF4_H200_FREE_TYPED_GPUS:?}" "${CF4_H100_FREE_TYPED_GPUS:?}" \
  "${CF4_A100_FREE_TYPED_GPUS:?}"
case "$CF4_GPU_MODE:$CF4_SUBMIT_PARTITION:$CF4_SUBMIT_GRES" in
  h200:h200:gpu:H200:1|h100:h100:gpu:H100:1|a100:a100:gpu:A100:1) ;;
  *) echo "unsupported recorded GPU mode: $CF4_GPU_MODE/$CF4_SUBMIT_PARTITION/$CF4_SUBMIT_GRES" >&2; exit 2 ;;
esac
for name in CF4_H200_FREE_TYPED_GPUS CF4_H100_FREE_TYPED_GPUS CF4_A100_FREE_TYPED_GPUS; do
  value="${!name}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$name must be an integer" >&2; exit 2; }
done
[[ "$SLURM_JOB_PARTITION" == "$CF4_SUBMIT_PARTITION" ]]
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_INFO" == *"$CF4_SUBMIT_GRES"* ]]

module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  CF4_MASTER_PLAN.md \
  CF4_R2_MALA_SHELL_DIRECTION_DIAG_PLAN_20261102.md \
  CF4_R2_BLOCK_MALA_SCREEN_RESULT_20261102.md \
  CF4_R2_BLOCK_MALA_SCREEN_ASTRA_RESULT_REVIEW_20261102.md \
  CF4_R2_BLOCK_MALA_SCREEN_ASTRA_REVIEW_20261102.md \
  CF4_R2_BLOCK_MALA_SCREEN_PLAN_20261102.md \
  scripts/cf4_r2_mala_shell_direction_diagnostic.py \
  scripts/run_cf4_r2_mala_shell_direction_diagnostic.sbatch \
  src/cf4_r2_block_mala.py \
  src/cf4_r2_block_elliptical_slice.py \
  tests/test_cf4_r2_block_mala.py \
  tests/test_cf4_r2_block_elliptical_slice.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_OUT_DIR="$CF4_R2_OUT_BASE/$SLURM_JOB_ID"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
GPU_MEMORY_LIMIT_BYTES=$("$PY" -c 'import jax; d=jax.devices()[0]; s=d.memory_stats(); assert d.platform == "gpu" and s and s.get("bytes_limit") is not None; print(int(s["bytes_limit"]))')
[[ "$GPU_MEMORY_LIMIT_BYTES" =~ ^[0-9]+$ ]] || {
  echo "cannot determine allocated JAX GPU memory limit" >&2
  exit 2
}
MIN_DEVICE_MEMORY_BYTES=52559495578
(( GPU_MEMORY_LIMIT_BYTES >= MIN_DEVICE_MEMORY_BYTES )) || {
  echo "allocated GPU lacks 20 percent headroom over measured peak" >&2
  exit 2
}
export CF4_GPU_MEMORY_LIMIT_BYTES="$GPU_MEMORY_LIMIT_BYTES"
"$PY" -m py_compile scripts/cf4_r2_mala_shell_direction_diagnostic.py
"$PY" -m unittest -v tests.test_cf4_r2_block_mala tests.test_cf4_r2_block_elliptical_slice
"$PY" tests/test_cf4_r2_conditional_optimizer_diagnostic.py -q
"$PY" tests/test_cf4_r2_vjp_geometry_metrics.py -q
"$PY" tests/test_cf4_r2_conditional_map_budget.py -q
"$PY" tests/test_cf4_r2_linked_point_conditioning_radius.py -q
exec timeout --signal=TERM --kill-after=30s 2700s \
  "$PY" scripts/cf4_r2_mala_shell_direction_diagnostic.py
