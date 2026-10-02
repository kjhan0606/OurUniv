#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_count_FP_own
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_R2_OUT_DIR:?}" "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
HEAD_REV=$(git rev-parse --verify HEAD^{commit})
EXPECTED_REV=$(git rev-parse --verify "${CF4_EXPECTED_COMMIT}^{commit}")
if [[ "$HEAD_REV" != "$EXPECTED_REV" ]]; then
    echo "source revision mismatch: expected=$CF4_EXPECTED_COMMIT actual=$HEAD_REV" >&2
    exit 2
fi
git diff --cached --quiet "$EXPECTED_REV" -- \
    src/cf4_r2_raw_live_mark.py src/cf4_r2_raw_volume_target.py \
    scripts/cf4_r2_count_fp_ownership_control.py \
    scripts/run_cf4_r2_count_fp_ownership_control.sbatch \
    tests/test_cf4_r2_raw_mixture_connection.py
git diff --quiet "$EXPECTED_REV" -- \
    src/cf4_r2_raw_live_mark.py src/cf4_r2_raw_volume_target.py \
    scripts/cf4_r2_count_fp_ownership_control.py \
    scripts/run_cf4_r2_count_fp_ownership_control.sbatch \
    tests/test_cf4_r2_raw_mixture_connection.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_observed_ray.py
"$PY" tests/test_cf4_r2_raw_mixture_connection.py
"$PY" tests/test_cf4_r2_raw_selected_fp.py
"$PY" tests/test_cf4_r2_shell_cdf_count.py
"$PY" tests/test_cf4_r2_raw_volume_target.py
exec "$PY" scripts/cf4_r2_count_fp_ownership_control.py
