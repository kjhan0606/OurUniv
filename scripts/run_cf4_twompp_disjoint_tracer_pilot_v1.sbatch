#!/usr/bin/env bash
#SBATCH --job-name=cf4-2mpp-input-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v1/logs/pilot-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v1/logs/pilot-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/bin/python3.13
program=config/cf4_twompp_disjoint_tracer_pilot_program_v1.json
source_file=src/cf4_twompp_disjoint_tracer_pilot.py
output=/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v1/pilot
program_sha=695f362b8942e911ebac5a942217bc75128cdbc45ca042e535b9bd2123517ee5
source_sha=695a21be6d5bb4cf51fee4f2c5c8d7ce5c1629dab7c744caf8a8960765fa84ec
expected_comment=cf4_twompp_disjoint_tracer_input_pilot_v1

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_twompp_disjoint_tracer_input_pilot_v1 ]]
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
[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 1024 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]]
[[ ! -e "$output" && ! -e "$(dirname "$output")/.pilot.staging" ]]

export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0
"$python" -I -P "$source_file" run-pilot \
  --program "$program" --output "$output" \
  --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$source_file" validate-pilot --directory "$output"
