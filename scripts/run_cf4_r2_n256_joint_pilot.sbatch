#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_N256joint
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
module purge
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src/cf4_r2_affine_force.py src/cf4_r2_resolution_target.py src/cf4_r2_chunked_volume_count.py src/cf4_r2_raw_volume_target.py src/cf4_r2_raw_live_mark.py src/cf4_r2_raw_selected_fp_jax.py src/cf4_r2_marked_tracer_jax.py src/cf4_r2_native_to_count_cells.py src/cf4_r2_shell_cdf_count.py src/cf4_r1_particle_forward.py scripts/cf4_r2_raw_field_profile.py scripts/cf4_r2_n256_joint_pilot.py scripts/cf4_r2_corrected_split_hmc.py scripts/cf4_r2_prior_split_hmc.py scripts/run_cf4_r2_n256_joint_pilot.sbatch
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_n256_joint_pilot.py
