#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-phaseb-smoke-v2
#SBATCH --partition=h100,h200,a100,a100_pcie,a40,a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24576M
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_datum_bearing_z0_phaseb_smoke_program_v2.json"
readonly implementation="$repo/src/cf4_datum_bearing_z0_phaseb_smoke.py"
readonly failure_record="$repo/config/cf4_datum_bearing_z0_phaseb_smoke_v1_failure_result_record.json"
readonly output=/gpfs/kjhan/CF4/z0_density/datum_bearing_phaseb_smoke_v2
readonly program_sha=3b0c23602b8ac8d850ea97dc52b80552d17b6e7bc58788edef6212ece338c351
readonly implementation_sha=c895b4ed9abe67685001c87d52c5ce50ecf34cf2021e1b32abe9c6b4518d7824
readonly failure_record_sha=f539f352c966b9fadc42c7ec672cdc0790ac4cb0d4133937dbb8e62b505edabe

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${OUTPUT:?OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$OUTPUT" == "$output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_correction ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ ! -e "$output" ]]
[[ ! -e "/gpfs/kjhan/CF4/z0_density/.datum_bearing_phaseb_smoke_v2.staging" ]]
[[ -f /gpfs/kjhan/CF4/z0_density/.datum_bearing_phaseb_smoke_v1.staging/result.json ]]

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
[[ "$(sha256sum "$failure_record" | awk '{print $1}')" == "$failure_record_sha" ]]

"$python" -I -P - "$program" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

program = json.loads(Path(sys.argv[1]).read_text())
assert sys.version.split()[0] == program["environment"]["python_version"]
for package in ("numpy", "jax", "jaxlib", "pmwd", "scipy"):
    assert metadata.version(package) == program["environment"][package]
for section in ("input_bindings", "source_bindings"):
    for record in program[section].values():
        actual = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        assert actual == record["sha256"], (record["path"], actual, record["sha256"])
assert program["authorization"]["actual_observational_field_inference"] is False
assert program["authorization"]["validation_seed_access"] is False
assert program["authorization"]["Phase_C_or_later"] is False
assert program["execution"]["maximum_submissions"] == 1
assert program["gates"]["rsd_mass_relative_error_max"] == 2e-13
assert program["gates"]["directional_gradient_relative_error_max"] == 2e-4
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 24576 ]]
command -v nvidia-smi >/dev/null
nvidia-smi -L | grep -q '^GPU '

export JAX_ENABLE_X64=True
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
  --output "$OUTPUT" \
  --implementation-commit "$EXPECTED_COMMIT"

"$python" -I -P "$implementation" validate --directory "$OUTPUT"
