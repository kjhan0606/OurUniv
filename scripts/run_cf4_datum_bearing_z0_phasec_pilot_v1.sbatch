#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-phasec
#SBATCH --partition=a40,a100,h100,h200,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.err
#SBATCH --open-mode=truncate

set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_program_v1.json"
readonly implementation="$repo/src/cf4_datum_bearing_z0_phasec_pilot.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v1
readonly program_sha=52633332a0aad094cfc088d195d92aa7129a2923d77c6246e1f71edf34b292cb
readonly implementation_sha=cc368f5207c1a6aae59e8a636ce42ff5197c08f9ab45eeddc2ee4664d1db5996

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$OUTPUT_ROOT" == "$output_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]

readonly cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]

readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]

"$python" -I -P - "$program" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

program = json.loads(Path(sys.argv[1]).read_text())
assert sys.version.split()[0] == program["environment"]["python_version"]
for package in ("numpy", "jax", "jaxlib", "pmwd", "scipy", "blackjax"):
    assert metadata.version(package) == program["environment"][package]
for section in ("input_bindings", "source_bindings"):
    for record in program[section].values():
        actual = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        assert actual == record["sha256"], (record["path"], actual, record["sha256"])
assert program["authorization"]["actual_observational_field_inference"] is False
assert program["authorization"]["actual_2Mpp_count_read"] is False
assert program["authorization"]["actual_CF4_velocity_datum_used"] is False
assert program["authorization"]["validation_seed_access"] is False
assert program["authorization"]["Phase_D_or_later"] is False
assert program["execution"]["array"] == "0-7"
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 10240 ]]
command -v nvidia-smi >/dev/null
nvidia-smi -L | grep -q '^GPU '

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P "$implementation" run \
  --program "$program" \
  --output-root "$OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --implementation-commit "$EXPECTED_COMMIT"

readonly seed=$((2026083000 + SLURM_ARRAY_TASK_ID))
readonly arms=AABBCCDD
readonly arm=${arms:SLURM_ARRAY_TASK_ID:1}
readonly task_dir=$(printf '%s/mock_%02d_seed_%d_arm_%s' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID" "$seed" "$arm")
"$python" -I -P "$implementation" validate-task --directory "$task_dir"
