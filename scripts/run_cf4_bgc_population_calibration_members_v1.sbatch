#!/usr/bin/env bash
#SBATCH --job-name=cf4-bgc-pop-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2048M
#SBATCH --time=01:00:00
#SBATCH --array=0-63%8
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1/logs/member-%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1/logs/member-%A_%a.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_bgc_population_calibration_program_v1.json
source_file=src/cf4_population_calibration.py
output_root=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1
members_root=${output_root}/members
program_sha=38ff44a066a55048f9d35ef93c64cce2f98702b4a785099087ffc97a698a00f0
source_sha=fe3b1dd38258b2ed28fd27a551287e14e797232f8f89da83ae1cc59c3309b2af

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_ARRAY_JOB_ID:?Slurm array job required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 64 ))
[[ -d "$output_root" ]]
[[ -d "$members_root" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]

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
assert execution["member_array"] == "0-63%8"
assert execution["maximum_member_array_submissions"] == 1
assert execution["member_requested_memory_MiB"] >= 1.2 * execution["member_expected_peak_memory_MiB"]
assert program["authorization"]["untouched_256_mock_validation"] is False
assert program["authorization"]["automatic_retry"] is False
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 2048 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE}" == 0 ]]

member_name=$(printf 'member-%02d' "$SLURM_ARRAY_TASK_ID")
output=${members_root}/${member_name}
[[ ! -e "$output" ]]
[[ ! -e "${members_root}/.${member_name}.staging" ]]

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
  --program "$program" \
  --mock-index "$SLURM_ARRAY_TASK_ID" \
  --output "$output" \
  --implementation-commit "$EXPECTED_COMMIT"

"$python" -I -P "$source_file" validate-member \
  --directory "$output" \
  --expected-index "$SLURM_ARRAY_TASK_ID"
