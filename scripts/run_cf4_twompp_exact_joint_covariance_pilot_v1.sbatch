#!/usr/bin/env bash
#SBATCH --job-name=cf4-2mpp-exact-joint-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3072M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/twompp_exact_joint_covariance_pilot_v1/logs/pilot-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/twompp_exact_joint_covariance_pilot_v1/logs/pilot-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
solver_python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_twompp_exact_joint_covariance_pilot_program_v1.json
source_file=src/cf4_twompp_exact_joint_covariance_pilot_v1.py
output_root=/gpfs/kjhan/CF4/kf_design/twompp_exact_joint_covariance_pilot_v1
output=$output_root/pilot
staging=$output_root/.pilot.${SLURM_JOB_ID}.staging
program_sha=4baddcb4d7e62a26d16e8ebfd9ecec8c09da5bec4aa8afb3bacd808b41776c08
source_sha=39086333ad6f73463cea29a32c541bce2d34bde1ed2cbf4a965ad02056276faa
expected_comment=cf4_twompp_exact_joint_covariance_pilot_v1

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_twompp_exact_joint_covariance_pilot_v1 ]]
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
[[ "$(readlink -f "$solver_python")" == "$solver_python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 3072 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ -d "$output_root" && -d "$output_root/logs" ]]
[[ ! -e "$staging" && ! -e "$output" ]]

export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0

"$solver_python" -I -P "$source_file" validate-program --program "$program"
"$solver_python" -I -P "$source_file" run-pilot \
  --program "$program" --staging "$staging" --output "$output" \
  --implementation-commit "$EXPECTED_COMMIT"
"$solver_python" -I -P "$source_file" validate-pilot --directory "$output"
