#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_joint_grad32
#SBATCH --partition=a100_pcie,a40,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- scripts/cf4_r2_joint_ic_gradient_control.py config/cf4_r2_common_cosmology_v1.json src/cf4_r1_particle_forward.py src/cf4_r2_continuous_tracer.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_r2_joint_ic_gradient_control.py
