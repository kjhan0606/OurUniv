#!/usr/bin/env bash
#SBATCH --job-name=cf4_r2_curv
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}" \
  "${CF4_SUBMIT_PARTITION:?record the selected partition}" \
  "${CF4_SUBMIT_GRES:?record the typed GRES request}"
case "$CF4_SUBMIT_PARTITION:$CF4_SUBMIT_GRES" in
  h200:gpu:H200:1|h100:gpu:H100:1|a100:gpu:A100:1) ;;
  *) echo "unsupported recorded GPU mode: $CF4_SUBMIT_PARTITION/$CF4_SUBMIT_GRES" >&2; exit 2 ;;
esac
[[ "$SLURM_JOB_PARTITION" == "$CF4_SUBMIT_PARTITION" ]]
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_INFO" == *"$CF4_SUBMIT_GRES"* ]]
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- \
  src/cf4_r2_curvature_range.py \
  scripts/cf4_r2_curvature_range_screen.py \
  scripts/run_cf4_r2_curvature_range_screen.sbatch \
  scripts/cf4_r2_conditional_map.py \
  tests/test_cf4_r2_curvature_range.py \
  CF4_R2_CURVATURE_RANGE_SCREEN_20261021.md CF4_MASTER_PLAN.md
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_curvature_range.py -q
"$PY" tests/test_cf4_r2_conditional_map_budget.py -q
"$PY" tests/test_cf4_r2_linked_point_conditioning_radius.py -q
"$PY" tests/test_cf4_r2_count_exposure.py -q
exec "$PY" scripts/cf4_r2_curvature_range_screen.py
