#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_v6_partial_map
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src scripts/cf4_r2_v6_partial_map.py scripts/cf4_r2_linked_fp_sparse_train.py scripts/run_cf4_r2_v6_partial_map.sbatch tests/test_cf4_r2_bounded_map_optimizer.py tests/test_cf4_r1_particle_forward.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/kjhan/CF4/cache/r2_linked_map
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r2_bounded_map_optimizer.py
/home/kjhan/miniconda3/envs/circle/bin/python3.11 tests/test_cf4_r1_particle_forward.py ParticleEntryTest.test_grid_integrals_fine_mesh
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_v6_partial_map.py
