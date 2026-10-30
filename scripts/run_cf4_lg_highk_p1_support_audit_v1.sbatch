#!/bin/bash
#SBATCH --job-name=cf4_highk_p1audit
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_p1_support_audit_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_p1_support_audit_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec /home/kjhan/miniconda3/bin/python -m cf4_lg_highk_p1_support_audit \
    --bank /gpfs/kjhan/CF4/recon/linear_cr/lowk_parent_posterior_bridge_v1/posterior_bank.npz \
    --bank-sha256 25ec02b9ddfa736c170c6c84332845f026790b1162f4ba14ba63fc2aa34d9e76 \
    --schedule /gpfs/kjhan/CF4/recon/linear_cr/lg_highk_joint_schedule_v1/schedule.npz \
    --schedule-sha256 10c8526f3508a908a14981636ca9f730480838b65ea4bbcd3186c0730f330ccd \
    --p1-result /gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v3/p1_result.json \
    --p1-result-sha256 050df61252ff038de9aec8fd6902dd42367b49c6ce1ac9782681158cf310cecd
