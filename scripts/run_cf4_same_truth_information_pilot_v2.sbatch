#!/usr/bin/env bash
#SBATCH --job-name=cf4-info-pilot-v2
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3072M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v2/logs/pilot-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v2/logs/pilot-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_same_truth_information_budget_program_v2.json
source_file=src/cf4_same_truth_information_budget_v2.py
output_root=/gpfs/kjhan/CF4/kf_design/bgc_same_truth_information_budget_v2
output=${output_root}/pilot/member-00
program_sha=5689193b61ce22c2170cd6b01f35d36bb8e22944ed2c4df758aa261ffc4a11bb
source_sha=ed0ab98780a7c2761ace06a63b5cb5eafa5411f7a6699e39e4f4187ad3ad3bbf
expected_comment=cf4_same_truth_information_v2_pilot

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_same_truth_information_pilot_v2 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]
[[ -d "$output_root" ]]
[[ -d "${output_root}/pilot" ]]
[[ ! -e "$output" ]]
[[ ! -e "${output_root}/pilot/.member-00.staging" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]
job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$job_record" == *"Comment=${expected_comment}"* ]]
head_commit=$(git -C "$repo" rev-parse HEAD)
upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$source_file" | awk '{print $1}')" == "$source_sha" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 3072 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]

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

"$python" -I -P "$source_file" run-member \
  --program "$program" --mock-index 0 --output "$output" \
  --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$source_file" validate-member --directory "$output" --expected-index 0
