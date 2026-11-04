#!/usr/bin/env bash
#SBATCH --job-name=cf4-r2-spatial-mcse
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_r2_spatial_mcse_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_r2_spatial_mcse_%j.err

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
output_root=/gpfs/kjhan/CF4/z0_density/r2_mcse_spatial_calibration_20261104
output_dir="$output_root/$SLURM_JOB_ID"

: "${SLURM_JOB_ID:?Slurm job id required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${CF4_EXPECTED_COMMIT:?CF4_EXPECTED_COMMIT must be explicitly exported}"
[[ "$CF4_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syn101 ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$CF4_EXPECTED_COMMIT" ]]
[[ "$(git -C "$repo" rev-parse '@{upstream}')" == "$CF4_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- \
    CF4_R2_SPATIAL_MCSE_CALIBRATION_PLAN_20261104.md \
    src/cf4_r2_mcse_spatial_calibration.py \
    scripts/cf4_r2_mcse_spatial_calibration.py \
    scripts/run_cf4_r2_mcse_spatial_calibration.sbatch \
    tests/test_cf4_r2_mcse_spatial_calibration.py)" ]]
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" == 2048 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ "$(readlink -f "$python")" == "$python" ]]

if ! git -C "$repo" diff --quiet "$CF4_EXPECTED_COMMIT" -- \
        CF4_R2_SPATIAL_MCSE_CALIBRATION_PLAN_20261104.md \
        src/cf4_r2_mcse_spatial_calibration.py \
        scripts/cf4_r2_mcse_spatial_calibration.py \
        scripts/run_cf4_r2_mcse_spatial_calibration.sbatch \
        tests/test_cf4_r2_mcse_spatial_calibration.py; then
    echo "spatial MCSE source differs from the pinned commit" >&2
    exit 2
fi

[[ ! -e "$output_dir" ]]
mkdir -p "$output_root"
mkdir "$output_dir"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python" -I -P -m unittest discover -s tests \
    -p test_cf4_r2_mcse_spatial_calibration.py -v
"$python" -I -P scripts/cf4_r2_mcse_spatial_calibration.py \
    --grid-size 16 --draws 512 --chains 4 --replicates 256 \
    --seed 20261104 --batch-sizes 32 64 --minimum-batches 8 \
    --correlation-lengths 0 1 2 --threshold 0.10 \
    --output "$output_dir/result.json"
printf '%s\n' \
    "job_id=$SLURM_JOB_ID" \
    "hostname=$host_name" \
    "commit=$CF4_EXPECTED_COMMIT" \
    "partition=${SLURM_JOB_PARTITION:?}" \
    "cpus=${SLURM_CPUS_PER_TASK:?}" \
    "mem_per_node=${SLURM_MEM_PER_NODE:?}" \
    "result=$output_dir/result.json"
