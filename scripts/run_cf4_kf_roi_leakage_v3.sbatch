#!/usr/bin/env bash
#SBATCH --job-name=cf4-kf-roi-leakage-v3-preflight
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/roi_leakage_v3/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/roi_leakage_v3/slurm-%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
design=config/cf4_kf_bin_manifest_design_v2.json
grant=config/cf4_kf_roi_leakage_execution_v3.json
output_root=/gpfs/kjhan/CF4/kf_design/roi_leakage_v3
preflight_output=${output_root}/preflight
design_sha=c2b1949b4fea26e96c79da57ff3d39aa654292f908ed414b038d094bf4303402
grant_sha=dc8f5b78e030fde11b8070e63273967cf63f03c8d1368e1a0425be92c444e6fa

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${MODE:?MODE must be explicitly exported as preflight}"
: "${OUTPUT:?OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$MODE" == preflight ]]
[[ "$OUTPUT" == "$preflight_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$output_root" ]]
[[ ! -e "$OUTPUT" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3; exit}')
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
[[ "$(sha256sum "$design" | awk '{print $1}')" == "$design_sha" ]]
[[ "$(sha256sum "$grant" | awk '{print $1}')" == "$grant_sha" ]]

[[ -x "$python" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
"$python" -I -P - <<'PY'
import importlib.metadata as metadata
import sys

assert sys.version.split()[0] == "3.11.15"
assert metadata.version("numpy") == "2.4.6"
assert metadata.version("scipy") == "1.17.1"
PY

[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 16 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 4096 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE}" == 0 ]]
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P src/cf4_kf_roi_leakage.py \
  --design "$design" \
  --execution-grant "$grant" \
  --output "$OUTPUT" \
  --mode preflight \
  --implementation-commit "$EXPECTED_COMMIT"

"$python" -I -P scripts/check_cf4_kf_roi_leakage_v3.py \
  --directory "$OUTPUT" \
  --design "$design" \
  --execution-grant "$grant" \
  --implementation-commit "$EXPECTED_COMMIT"
