#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_rawlive
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
module purge
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src/cf4_r2_raw_live_mark.py src/cf4_r2_raw_selected_fp_jax.py src/cf4_r2_marked_tracer_jax.py src/cf4_r2_native_to_count_cells.py scripts/cf4_r2_raw_live_check.py scripts/run_cf4_r2_raw_live_check.sbatch
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_raw_live_check.py
