#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_N256GL2
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}"
: "${CF4_EXPECTED_COMMIT:?}"
: "${CF4_R2_CHAIN_LABEL:?set to a or b}"
: "${CF4_R2_INITIAL_CHECKPOINT:?set corrected-target checkpoint}"
: "${CF4_R2_OUT_DIR:?set unique output directory}"
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- scripts/cf4_r2_force_pair_comparison.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_EXACT_CHAIN_STEPS=1
export CF4_R2_CHAIN_INTEGRATION_STEPS=1
export CF4_R2_FUNDAMENTAL_MASS=6000
export CF4_R2_CHAIN_SEED=2026100401
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" -m unittest discover -s tests -p test_cf4_r2_prior_split_hmc.py -q
exec "$PY" scripts/cf4_r2_force_pair_comparison.py
