#!/usr/bin/env bash
#SBATCH --job-name=cf4-2mpp-joint-info-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3072M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/twompp_joint_information_budget_pilot_v1/logs/pilot-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/twompp_joint_information_budget_pilot_v1/logs/pilot-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
selection_python=/home/kjhan/miniconda3/bin/python3.13
solver_python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_twompp_joint_information_budget_pilot_program_v1.json
source_file=src/cf4_twompp_joint_information_budget_pilot_v1.py
stage=/gpfs/kjhan/CF4/kf_design/twompp_joint_information_budget_pilot_v1/.pilot.staging
output=/gpfs/kjhan/CF4/kf_design/twompp_joint_information_budget_pilot_v1/pilot
program_sha=7ffa68d6996572ec9088bd349e4adf7809d3678635ebab69fecba23c87a1b7f2
source_sha=d318031ef47a4063671f77d564565afd63ca97bb0920580cb779f2004e83df22
expected_comment=cf4_twompp_joint_information_budget_pilot_v1

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_twompp_joint_information_budget_pilot_v1 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* ]]
[[ "$host_name" != syn101 && "$host_name" != syn101.* ]]
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
[[ "$(readlink -f "$selection_python")" == "$selection_python" ]]
[[ "$(readlink -f "$solver_python")" == "$solver_python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 3072 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ ! -e "$stage" && ! -e "$output" ]]

export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0

"$selection_python" -I -P "$source_file" build-selection \
  --program "$program" --stage "$stage"
"$solver_python" -I -P "$source_file" run-pilot \
  --program "$program" --stage "$stage" --output "$output" \
  --implementation-commit "$EXPECTED_COMMIT"
"$solver_python" -I -P "$source_file" validate-pilot --directory "$output"
