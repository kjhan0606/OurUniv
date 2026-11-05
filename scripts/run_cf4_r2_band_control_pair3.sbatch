#!/usr/bin/env bash
#SBATCH --job-name=cf4-band-p3
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}" \
  "${CF4_GPU_MODE:?}" "${CF4_H200_FREE_TYPED_GPUS:?}" \
  "${CF4_H100_FREE_TYPED_GPUS:?}" "${CF4_A100_FREE_TYPED_GPUS:?}" \
  "${CF4_BAND_CONTROL_START_PAIR:?}" "${CF4_BAND_CONTROL_PRIOR_RESULT:?}"
[[ "$CF4_BAND_CONTROL_START_PAIR" == 3 ]]
for count in "$CF4_H200_FREE_TYPED_GPUS" "$CF4_H100_FREE_TYPED_GPUS" \
             "$CF4_A100_FREE_TYPED_GPUS"; do
  [[ "$count" =~ ^[0-9]+$ ]]
done
# These snapshots mean GPUs eligible to this account after partition/QoS caps.
case "$CF4_GPU_MODE" in
  h200)
    (( CF4_H200_FREE_TYPED_GPUS > 0 ))
    ;;
  h100)
    (( CF4_H200_FREE_TYPED_GPUS == 0 && CF4_H100_FREE_TYPED_GPUS > 0 ))
    ;;
  a100)
    (( CF4_H200_FREE_TYPED_GPUS == 0 && CF4_H100_FREE_TYPED_GPUS == 0 \
       && CF4_A100_FREE_TYPED_GPUS > 0 ))
    ;;
  *) echo "unsupported GPU mode: $CF4_GPU_MODE" >&2; exit 1 ;;
esac
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  scripts/cf4_r2_band_control.py \
  scripts/run_cf4_r2_band_control_pair3.sbatch \
  tests/test_cf4_r2_band_metric.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
PAIR3_START_LOCAL=$CF4_BAND_CONTROL_START_PAIR
PRIOR_RESULT_LOCAL=$CF4_BAND_CONTROL_PRIOR_RESULT
unset CF4_BAND_CONTROL_START_PAIR CF4_BAND_CONTROL_PRIOR_RESULT
"$PY" tests/test_cf4_r2_band_metric.py -q
exec env CF4_BAND_CONTROL_START_PAIR="$PAIR3_START_LOCAL" \
  CF4_BAND_CONTROL_PRIOR_RESULT="$PRIOR_RESULT_LOCAL" \
  "$PY" scripts/cf4_r2_band_control.py
