#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_fp_restart
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
module purge
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
git diff --exit-code "$CF4_EXPECTED_COMMIT" -- src scripts/cf4_r2_fp_restart_check.py scripts/cf4_r2_raw_source_mixture.py scripts/cf4_r2_linked_fp_sparse_train.py scripts/run_cf4_r2_fp_restart_check.sbatch tests/test_cf4_r2_linked_singleton_jax.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CF4_R2_FP_SCIENCE_STATE:-}" ]]; then
    /home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_linked_singleton_jax.py
fi
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_fp_restart_check.py
