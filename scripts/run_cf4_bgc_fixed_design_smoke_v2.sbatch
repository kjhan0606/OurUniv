#!/usr/bin/env bash
#SBATCH --job-name=cf4-bgc-fd-smoke-v2
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1536M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/bgc_fixed_design_smoke_v2/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/bgc_fixed_design_smoke_v2/slurm-%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
config=config/cf4_bgc_fixed_design_smoke_execution_v2.json
catalog=data/cf4_clean.npz
manifest=config/cf4_kf_bin_manifest_v1.json
predecessor_record=config/cf4_bgc_fixed_design_smoke_v1_result.json
output_root=/gpfs/kjhan/CF4/kf_design/bgc_fixed_design_smoke_v2
final_output=${output_root}/smoke
config_sha=ebbd80497ccb5175de4581fccfc1216fcebd61cf5e32b5a8e9d60f541f07b446
predecessor_record_sha=d08faa7ac968aa8226f61a6282ea8463bcb5437274e9a8593e5ad734e1588f87

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${MODE:?MODE must be explicitly exported as fixed_design_theta_correction_smoke}"
: "${OUTPUT:?OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$MODE" == fixed_design_theta_correction_smoke ]]
[[ "$OUTPUT" == "$final_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$output_root" ]]
[[ ! -e "$OUTPUT" ]]
[[ ! -e "${output_root}/.smoke.staging" ]]

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
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$config_sha" ]]
[[ "$(sha256sum "$predecessor_record" | awk '{print $1}')" == "$predecessor_record_sha" ]]

"$python" -I -P - "$config" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
assert sys.version.split()[0] == config["environment"]["python_version"]
for package in ("numpy", "jax", "jaxlib", "pmwd", "scipy"):
    assert metadata.version(package) == config["environment"][package]
for section in ("input_bindings", "source_bindings"):
    for record in config[section].values():
        expected = record.get("sha256", record.get("file_sha256"))
        actual = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        assert actual == expected, (record["path"], actual, expected)
execution = config["execution"]
assert execution["maximum_correction_submissions"] == 1
assert execution["maximum_correction_executions"] == 1
assert config["authorization"]["new_unique_development_truth_seed_authorized"] is False
assert config["authorization"]["additional_retry_authorized"] is False
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 1536 ]]
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

"$python" -I -P src/cf4_bgc_fixed_design_smoke.py run \
  --catalog "$catalog" \
  --bin-manifest "$manifest" \
  --output "$OUTPUT" \
  --implementation-commit "$EXPECTED_COMMIT"

"$python" -I -P src/cf4_bgc_fixed_design_smoke.py validate \
  --directory "$OUTPUT"
