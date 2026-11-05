#!/usr/bin/env bash
#SBATCH --job-name=cf4-r2-n256-profile
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_r2_n256_profile_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_r2_n256_profile_%j.err

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
output_root=/gpfs/kjhan/CF4/z0_density/r2_n256_report_profile_20261105
training_split=/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v6/split.npz

: "${SLURM_JOB_ID:?Slurm job id required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${CF4_EXPECTED_COMMIT:?CF4_EXPECTED_COMMIT must be explicitly exported}"
expected_commit=$CF4_EXPECTED_COMMIT
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$(hostname -s)" != syntax && "$(hostname -s)" != syn101 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$expected_commit" ]]
[[ "$(git -C "$repo" rev-parse '@{upstream}')" == "$expected_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- \
    CF4_R2_N256_REPORT_RESOURCE_PROFILE_PLAN_20261105.md \
    scripts/cf4_r2_n256_report_resource_profile.py \
    scripts/run_cf4_r2_n256_report_resource_profile.sbatch \
    tests/test_cf4_r2_n256_report_resource_profile.py)" ]]
git -C "$repo" diff --quiet "$expected_commit" -- \
    CF4_R2_N256_REPORT_RESOURCE_PROFILE_PLAN_20261105.md \
    scripts/cf4_r2_n256_report_resource_profile.py \
    scripts/run_cf4_r2_n256_report_resource_profile.sbatch \
    tests/test_cf4_r2_n256_report_resource_profile.py
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" == 28672 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
[[ -r "$training_split" && -d /gpfs/kjhan/CF4/logs ]]

mkdir -p "$output_root"
python -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; print(f"GPFS free bytes at job start: {free}"); sys.exit(0 if free >= 55*(1024**3) else 3)' /gpfs/kjhan/CF4
job_root="$output_root/$SLURM_JOB_ID"
[[ ! -e "$job_root" ]]
mkdir "$job_root"

export CF4_EXPECTED_COMMIT="$expected_commit"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --exact --ntasks=1 --cpus-per-task=1 "$python" -I -P -m unittest \
    discover -s tests -p test_cf4_r2_n256_report_resource_profile.py -v

SECONDS=0
srun --exact --ntasks=1 --cpus-per-task=1 "$python" -I -P \
    scripts/cf4_r2_n256_report_resource_profile.py build \
    --job-root "$job_root" --training-split "$training_split" \
    --grid-size 256 --draws 512 --batch-sizes 32 64 \
    --minimum-batches 8 --count-grid-size 128 --population-count 6
generation_seconds=$SECONDS

SECONDS=0
srun --exact --ntasks=1 --cpus-per-task=1 "$python" -I -P \
    scripts/cf4_r2_four_chain_field_mcse_report.py \
    "$job_root/fixtures/chain_0/result.json" \
    "$job_root/fixtures/chain_1/result.json" \
    "$job_root/fixtures/chain_2/result.json" \
    "$job_root/fixtures/chain_3/result.json" \
    --training-split "$training_split" \
    --output "$job_root/raw_synthetic_report.json"
report_seconds=$SECONDS

srun --exact --ntasks=1 --cpus-per-task=1 "$python" -I -P \
    scripts/cf4_r2_n256_report_resource_profile.py finalize \
    --job-root "$job_root" \
    --raw-report "$job_root/raw_synthetic_report.json" \
    --training-split "$training_split" --grid-size 256 --draws 512 \
    --batch-sizes 32 64 --generation-seconds "$generation_seconds" \
    --report-seconds "$report_seconds" --requested-mem-gib 28 \
    --expected-peak-gib 21 --count-grid-size 128 --population-count 6

printf '%s\n' \
    "job_id=$SLURM_JOB_ID" \
    "hostname=$(hostname -s)" \
    "commit=$expected_commit" \
    "partition=${SLURM_JOB_PARTITION:?}" \
    "cpus=${SLURM_CPUS_PER_TASK:?}" \
    "mem_per_node=${SLURM_MEM_PER_NODE:?}" \
    "resource_profile=$job_root/resource_profile.json"
