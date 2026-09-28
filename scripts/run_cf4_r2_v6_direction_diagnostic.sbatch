#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_v6_direction_diag
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src scripts/cf4_r2_v6_partial_map.py scripts/cf4_r2_linked_fp_sparse_train.py scripts/run_cf4_r2_v6_direction_diagnostic.sbatch
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/kjhan/CF4/cache/r2_linked_map
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_DIRECTION_DIAG=1
export CF4_R2_OUT_DIR=/gpfs/kjhan/CF4/z0_density/r2_v6_map_direction_diag_v1
export CF4_R2_RESTART=/gpfs/kjhan/CF4/z0_density/r2_v6_partial_map_v2/accepted_checkpoint.npz
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_v6_partial_map.py
