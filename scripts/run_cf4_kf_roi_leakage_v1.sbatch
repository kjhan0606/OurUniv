#!/usr/bin/env bash
#SBATCH --job-name=cf4-kf-roi-leakage-v1
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/roi_leakage_v1/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/roi_leakage_v1/slurm-%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
design=config/cf4_kf_bin_manifest_design_v1.json
grant=config/cf4_kf_roi_leakage_execution_v1.json
output_root=/gpfs/kjhan/CF4/kf_design/roi_leakage_v1
preflight_output=${output_root}/preflight
production_output=${output_root}/production
receipt=${output_root}/preflight_MaxRSS_receipt.json

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${MODE:?MODE must be explicitly exported as preflight or production}"
: "${OUTPUT:?OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"

[[ "$MODE" == preflight || "$MODE" == production ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$output_root" ]]
[[ ! -e "$OUTPUT" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3; exit}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* ]]
job_state=$(squeue -h -j "$SLURM_JOB_ID" -o '%T')
[[ -n "$job_state" ]]

head_commit=$(git -C "$repo" rev-parse HEAD)
upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]

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

if [[ "$MODE" == preflight ]]; then
  [[ "$OUTPUT" == "$preflight_output" ]]
  [[ "${SLURM_MEM_PER_NODE:-0}" -ge 4096 ]]
  "$python" -I -P src/cf4_kf_roi_leakage.py \
    --design "$design" \
    --output "$OUTPUT" \
    --mode preflight \
    --implementation-commit "$EXPECTED_COMMIT"
else
  : "${PREFLIGHT_RESULT:?PREFLIGHT_RESULT must be explicitly exported}"
  : "${PREFLIGHT_RESULT_SHA256:?PREFLIGHT_RESULT_SHA256 must be exported}"
  : "${PREFLIGHT_MAXRSS_RECEIPT:?PREFLIGHT_MAXRSS_RECEIPT must be exported}"
  : "${PREFLIGHT_MAXRSS_RECEIPT_SHA256:?receipt SHA256 must be exported}"
  : "${REQUIRED_MEM_MIB:?REQUIRED_MEM_MIB must be explicitly exported}"
  [[ "$OUTPUT" == "$production_output" ]]
  [[ "$PREFLIGHT_RESULT" == "$preflight_output/result.json" ]]
  [[ "$PREFLIGHT_MAXRSS_RECEIPT" == "$receipt" ]]
  [[ "$PREFLIGHT_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ "$PREFLIGHT_MAXRSS_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ "$(sha256sum "$PREFLIGHT_RESULT" | awk '{print $1}')" == "$PREFLIGHT_RESULT_SHA256" ]]
  [[ "$(sha256sum "$receipt" | awk '{print $1}')" == "$PREFLIGHT_MAXRSS_RECEIPT_SHA256" ]]
  calculated_mem_mib=$("$python" -I -P - "$receipt" "$PREFLIGHT_RESULT_SHA256" <<'PY'
import json
import math
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1])
expected_result_sha = sys.argv[2]
record = json.loads(receipt_path.read_text(encoding="utf-8"))
if record.get("schema") != "ouruniv-cf4-kf-roi-leakage-maxrss-receipt-v1":
    raise SystemExit("bad MaxRSS receipt schema")
if record.get("status") != "PASS":
    raise SystemExit("MaxRSS receipt is not PASS")
if record.get("preflight_result_sha256") != expected_result_sha:
    raise SystemExit("MaxRSS receipt/result binding mismatch")
maxrss_mib = record.get("MaxRSS_MiB")
if not isinstance(maxrss_mib, (int, float)) or not math.isfinite(maxrss_mib) or maxrss_mib <= 0:
    raise SystemExit("invalid MaxRSS_MiB")
print(max(1024, math.ceil(maxrss_mib * 1.2)))
PY
)
  [[ "$REQUIRED_MEM_MIB" =~ ^[0-9]+$ ]]
  [[ "$REQUIRED_MEM_MIB" -eq "$calculated_mem_mib" ]]
  [[ "${SLURM_MEM_PER_NODE:-0}" -ge "$REQUIRED_MEM_MIB" ]]
  "$python" -I -P src/cf4_kf_roi_leakage.py \
    --design "$design" \
    --output "$OUTPUT" \
    --mode production \
    --preflight-result "$PREFLIGHT_RESULT" \
    --preflight-result-sha256 "$PREFLIGHT_RESULT_SHA256" \
    --implementation-commit "$EXPECTED_COMMIT"
fi

"$python" -I -P scripts/check_cf4_kf_roi_leakage_v1.py \
  --directory "$OUTPUT" \
  --design "$design" \
  --execution-grant "$grant" \
  --expected-mode "$MODE" \
  --implementation-commit "$EXPECTED_COMMIT"
