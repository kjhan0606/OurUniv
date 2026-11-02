#!/usr/bin/env bash
#SBATCH --job-name=cf4_r2_hmc_jumps
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_BASE:?}" \
  "${CF4_HMC_START:?}" "${CF4_GPU_MODE:?}" \
  "${CF4_SUBMIT_PARTITION:?}" "${CF4_SUBMIT_GRES:?}" \
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
if (( CF4_H200_FREE_TYPED_GPUS > 0 )); then
  [[ "$CF4_GPU_MODE" == h200 ]] || { echo "H200 was free and must be preferred" >&2; exit 2; }
elif (( CF4_H100_FREE_TYPED_GPUS > 0 )); then
  [[ "$CF4_GPU_MODE" == h100 ]] || { echo "H200 busy; free H100 must be preferred" >&2; exit 2; }
elif (( CF4_A100_FREE_TYPED_GPUS > 0 )); then
  [[ "$CF4_GPU_MODE" == a100 ]] || { echo "H200/H100 busy; free A100 must be selected" >&2; exit 2; }
else
  echo "no free typed H200/H100/A100 GPU in pre-submit snapshot" >&2
  exit 2
fi
[[ "$SLURM_JOB_PARTITION" == "$CF4_SUBMIT_PARTITION" ]]
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_INFO" == *"$CF4_SUBMIT_GRES"* ]]
[[ -s "$CF4_HMC_START" && -s "${CF4_HMC_START%/*}/result.json" ]]

module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  CF4_MASTER_PLAN.md \
  CF4_R2_HMC_RETAINED_BLOCK_JUMPS_PLAN_20261102.md \
  scripts/cf4_r2_hmc_transport.py \
  scripts/run_cf4_r2_hmc_retained_block_jumps.sbatch \
  tests/test_cf4_r2_hmc_transport.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_HMC_END_INDEX=62 CF4_HMC_LENGTHS=8,12
export CF4_HMC_MAX_EVALUATIONS=25 CF4_HMC_APP_SECONDS=14400
export CF4_R2_OUT_DIR="$CF4_R2_OUT_BASE/$SLURM_JOB_ID"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
GPU_MEMORY_LIMIT_BYTES=$("$PY" -c 'import jax; d=jax.devices()[0]; s=d.memory_stats(); assert d.platform == "gpu" and s and s.get("bytes_limit") is not None; print(int(s["bytes_limit"]))')
[[ "$GPU_MEMORY_LIMIT_BYTES" =~ ^[0-9]+$ ]] || {
  echo "cannot determine allocated JAX GPU memory limit" >&2
  exit 2
}
MIN_DEVICE_MEMORY_BYTES=39826427463
(( GPU_MEMORY_LIMIT_BYTES >= MIN_DEVICE_MEMORY_BYTES )) || {
  echo "allocated GPU lacks 20 percent headroom over measured exact-GL2 peak" >&2
  exit 2
}
export CF4_GPU_MODE CF4_GPU_MEMORY_LIMIT_BYTES="$GPU_MEMORY_LIMIT_BYTES"
export CF4_SUBMIT_PARTITION CF4_SUBMIT_GRES
export CF4_H200_FREE_TYPED_GPUS CF4_H100_FREE_TYPED_GPUS CF4_A100_FREE_TYPED_GPUS
"$PY" -m py_compile scripts/cf4_r2_hmc_transport.py
"$PY" tests/test_cf4_r2_hmc_transport.py -q
"$PY" tests/test_cf4_r2_prior_split_hmc.py -q
"$PY" tests/test_cf4_r2_conditional_map_budget.py -q
"$PY" tests/test_cf4_r2_linked_point_conditioning_radius.py -q
exec timeout --signal=TERM --kill-after=30s 14400s \
  "$PY" scripts/cf4_r2_hmc_transport.py
