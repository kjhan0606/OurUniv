#!/usr/bin/env bash
#SBATCH --job-name=cf4-r2-joint-mcse
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_r2_joint_mcse_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_r2_joint_mcse_%j.err

set -euo pipefail
set -x

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
output_root=/gpfs/kjhan/CF4/z0_density/r2_joint_field_mcse_stress_20261105

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
    CF4_R2_JOINT_FIELD_MCSE_STRESS_PLAN_20261105.md \
    src/cf4_r2_joint_field_mcse_stress.py \
    scripts/cf4_r2_joint_field_mcse_stress.py \
    scripts/run_cf4_r2_joint_field_mcse_stress.sbatch \
    tests/test_cf4_r2_joint_field_mcse_stress.py)" ]]
if ! git -C "$repo" diff --quiet "$CF4_EXPECTED_COMMIT" -- \
        CF4_R2_JOINT_FIELD_MCSE_STRESS_PLAN_20261105.md \
        src/cf4_r2_joint_field_mcse_stress.py \
        scripts/cf4_r2_joint_field_mcse_stress.py \
        scripts/run_cf4_r2_joint_field_mcse_stress.sbatch \
        tests/test_cf4_r2_joint_field_mcse_stress.py; then
    echo "joint field-MCSE stress source differs from pinned commit" >&2
    exit 2
fi
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" == 2048 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ "$(readlink -f "$python")" == "$python" ]]

output_dir="$output_root/$SLURM_JOB_ID"
[[ ! -e "$output_dir" ]]
mkdir -p "$output_root"
mkdir "$output_dir"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The tested files run first; any failure stops before the fixed oracle or
# stress replicates. All work in this batch job remains CPU-only and bounded.
for test_name in \
        test_cf4_r2_joint_field_mcse_stress.py \
        test_cf4_r2_mcse_spatial_calibration.py \
        test_cf4_r2_field_batch_mcse.py \
        test_cf4_r2_cross_chain_field_diagnostics.py \
        test_cf4_r2_four_chain_field_mcse.py; do
    # -I ignores PYTHONPATH, which these source-tree tests require. User-site
    # packages remain disabled above, and -P avoids implicit CWD imports.
    "$python" -P -m unittest discover -s tests -p "$test_name" -v
done

"$python" -P scripts/cf4_r2_joint_field_mcse_stress.py \
    --output-dir "$output_dir" \
    --seed 20261105 \
    --oracle-seed 20261106 \
    --oracle-histories 250000 \
    --oracle-chunk-size 8192 \
    --replicates 256
printf '%s\n' \
    "job_id=$SLURM_JOB_ID" \
    "hostname=$host_name" \
    "commit=$CF4_EXPECTED_COMMIT" \
    "partition=${SLURM_JOB_PARTITION:?}" \
    "cpus=${SLURM_CPUS_PER_TASK:?}" \
    "mem_per_node=${SLURM_MEM_PER_NODE:?}" \
    "result=$output_dir/result.json" \
    "oracle=$output_dir/oracle.json" \
    "replicates=$output_dir/replicates.jsonl"
