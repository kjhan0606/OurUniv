#!/usr/bin/env bash
#SBATCH --job-name=cf4-kf-cal-smoke-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/cf4_calibration_smoke_v1/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/cf4_calibration_smoke_v1/slurm-%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
config=config/cf4_kf_calibration_smoke_execution_v1.json
manifest=config/cf4_kf_bin_manifest_v1.json
output_root=/gpfs/kjhan/CF4/kf_design/cf4_calibration_smoke_v1
final_output=${output_root}/smoke
config_sha=6e8c1829b774780de6071d82ac19623ae11a617bede290c64b1ec1bcb474eee1
manifest_sha=d76e1a6dbe0b60c5c0f50a2e3c619b041f72c6b686a2fb3d977455b1c874b302

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${MODE:?MODE must be explicitly exported as implementation_smoke}"
: "${OUTPUT:?OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$MODE" == implementation_smoke ]]
[[ "$OUTPUT" == "$final_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$output_root" ]]
[[ ! -e "$OUTPUT" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]
job_state=$(squeue -h -j "$SLURM_JOB_ID" -o '%T')
[[ -n "$job_state" ]]

head_commit=$(git -C "$repo" rev-parse HEAD)
upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$config_sha" ]]
[[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$manifest_sha" ]]

[[ -x "$python" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
"$python" -I -P - <<'PY'
import importlib.metadata as metadata
import sys

assert sys.version.split()[0] == "3.11.15"
assert metadata.version("numpy") == "2.4.6"
PY

[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 512 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE}" == 0 ]]
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

staging=${output_root}/.smoke.staging-${SLURM_JOB_ID}
[[ ! -e "$staging" ]]
mkdir -- "$staging"

"$python" -I -P scripts/cf4_kf_calibration_smoke.py run \
  --config "$config" \
  --manifest "$manifest" \
  --output-directory "$staging" \
  --implementation-commit "$EXPECTED_COMMIT"

"$python" -I -P scripts/cf4_kf_calibration_smoke.py validate \
  --directory "$staging" \
  --config "$config" \
  --manifest "$manifest"

mv -- "$staging" "$OUTPUT"

"$python" -I -P scripts/cf4_kf_calibration_smoke.py validate \
  --directory "$OUTPUT" \
  --config "$config" \
  --manifest "$manifest"
