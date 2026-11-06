#!/usr/bin/env bash
#SBATCH --job-name=cf4-r2-cond-fp
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_r2_cond_fp_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_r2_cond_fp_%j.err
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$CF4_EXPECTED_COMMIT" ]]
[[ "$(hostname -s)" != syntax && "$(hostname -s)" != syn101 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
git -C "$repo" diff --exit-code "$CF4_EXPECTED_COMMIT" -- \
    src/cf4_r2_conditional_fp.py src/cf4_r2_raw_selected_fp.py \
    scripts/cf4_r2_conditional_fp_study.py tests/test_cf4_r2_conditional_fp.py
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$python" -P -m unittest discover -s tests -p test_cf4_r2_conditional_fp.py -v
"$python" -P -m unittest discover -s tests -p test_cf4_r2_raw_selected_fp.py -v
study_root=/gpfs/kjhan/CF4/z0_density/r2_conditional_fp_component_20261107
mkdir -p "$study_root"
"$python" -P scripts/cf4_r2_conditional_fp_study.py --output-dir "$study_root/$SLURM_JOB_ID"
