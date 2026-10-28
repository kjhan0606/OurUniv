#!/usr/bin/env bash
#SBATCH --job-name=cf4_r2_valueprobe
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=02:30:00
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
  src/cf4_r2_scalar_primal.py \
  scripts/cf4_r2_conditional_map.py \
  scripts/cf4_r2_scalar_value_cost_probe.py \
  scripts/run_cf4_r2_scalar_value_cost_probe.sbatch \
  tests/test_cf4_r2_scalar_primal.py CF4_MASTER_PLAN.md
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_OUT_DIR="$CF4_R2_OUT_BASE/$SLURM_JOB_ID"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
exec "$PY" scripts/cf4_r2_scalar_value_cost_probe.py
