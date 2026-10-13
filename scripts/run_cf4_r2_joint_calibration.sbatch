#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_jointcal
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:16:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
module purge
cd /home/kjhan/BACKUP/CF4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_joint_calibration.py
"$PY" tests/test_cf4_r2_nonfp_marks.py
case "${1:-}" in
    '') "$PY" scripts/cf4_r2_fp_group_marginal_control.py --joint-calibration-cache ;;
    --reuse-cache) test -f /gpfs/kjhan/CF4/z0_density/r2_joint_calibration_cache_v1/manifest.json ;;
    *) echo 'only optional argument is --reuse-cache' >&2; exit 2 ;;
esac
exec "$PY" scripts/cf4_r2_joint_calibration.py
