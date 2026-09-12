#!/usr/bin/env bash
#SBATCH --job-name=cf4-info-agg-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3072M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v1/logs/aggregate-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v1/logs/aggregate-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_same_truth_information_budget_program_v1.json
source_file=src/cf4_same_truth_information_budget.py
output_root=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v1
members_root=${output_root}/members
output=${output_root}/aggregate
program_sha=862bf6c125c228b6e9be9e8b4178c976d15eb4bb015cd506d7a066c8c0e9682a
source_sha=d694505367807bea83e0e7ed0686c2b1b89ac315546f04fdfbaaaa6b328db3cc

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${MEMBER_IMPLEMENTATION_COMMIT:?MEMBER_IMPLEMENTATION_COMMIT must be explicitly exported}"
: "${MEMBER_ARRAY_JOB_ID:?MEMBER_ARRAY_JOB_ID must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_same_truth_information_afterok_aggregate_v1 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$MEMBER_IMPLEMENTATION_COMMIT" == "$EXPECTED_COMMIT" ]]
[[ "$MEMBER_ARRAY_JOB_ID" =~ ^[0-9]+$ ]]
[[ -d "$output_root" ]]
[[ -d "$members_root" ]]
[[ ! -e "$output" ]]
[[ ! -e "${output_root}/.aggregate.staging" ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]
job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
expected_comment=cf4_same_truth_information_afterok_${MEMBER_ARRAY_JOB_ID}
[[ "$job_record" == *"Comment=${expected_comment}"* ]]

head_commit=$(git -C "$repo" rev-parse HEAD)
upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$source_file" | awk '{print $1}')" == "$source_sha" ]]

"$python" -I -P - "$program" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

program = json.loads(Path(sys.argv[1]).read_text())
assert sys.version.split()[0] == program["environment"]["python_version"]
for package in ("numpy", "jax", "jaxlib", "pmwd", "scipy"):
    assert metadata.version(package) == program["environment"][package]
execution = program["execution"]
assert execution["maximum_aggregate_submissions"] == 1
assert execution["aggregate_requested_memory_MiB"] >= 1.2 * execution["aggregate_expected_peak_memory_MiB"]
assert program["design"]["covariance_only"] is True
assert program["authorization"]["automatic_follow_on_after_aggregate"] is False
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 3072 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE}" == 0 ]]

export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P "$source_file" aggregate \
  --program "$program" \
  --members-root "$members_root" \
  --output "$output" \
  --member-implementation-commit "$MEMBER_IMPLEMENTATION_COMMIT" \
  --aggregation-runtime-commit "$EXPECTED_COMMIT"

"$python" -I -P "$source_file" validate-aggregate --directory "$output"
