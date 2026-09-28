#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_split_pilot
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src scripts/cf4_r2_v6_partial_map.py scripts/cf4_r2_linked_fp_sparse_train.py scripts/cf4_r2_prior_split_hmc.py scripts/cf4_r2_split_sampling_run.py scripts/run_cf4_r2_split_sampling.sbatch tests/test_cf4_r2_prior_split_hmc.py tests/test_cf4_r2_bounded_map_optimizer.py tests/test_cf4_r2_linked_singleton_jax.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/kjhan/CF4/cache/r2_linked_map
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CF4_R2_SPLIT_SAMPLING=1 CF4_R2_INCLUDE_GROUPED_SINGLE_MARK=1
export CF4_R2_COUNT_INTEGRATION=shell_cdf CF4_R2_CDF_ORDER=4 CF4_R2_CDF_SEGMENTS=32
export CF4_R2_SECONDS_CAP=4500
unset CF4_R2_NUISANCE_ONLY CF4_R2_NUISANCE_METRIC_REPORT CF4_R2_FP_COHORT_REFERENCE
unset CF4_R2_DIRECTION_DIAG CF4_R2_CURVATURE_DIAG CF4_R2_RATE_WARM_START
export CF4_R2_RESTART=/gpfs/kjhan/CF4/z0_density/r2_v6_single_mark_map_v1/accepted_checkpoint.npz
export CF4_R2_SAMPLER_METRIC_REPORT=/gpfs/kjhan/CF4/z0_density/r2_v6_joint_secant_map_v3/result.json
/home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_prior_split_hmc.py
/home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_bounded_map_optimizer.py
/home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_linked_singleton_jax.py
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_v6_partial_map.py
