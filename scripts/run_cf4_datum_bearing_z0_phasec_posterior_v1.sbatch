#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-posterior-v1
#SBATCH --partition=a40
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240M
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_posterior_v1.json"
readonly implementation="$repo/src/cf4_datum_bearing_z0_phasec_posterior_v1.py"
readonly device_program="$repo/config/cf4_phasec_same_gpu_repeatability_diagnostic_v1.json"
readonly device_implementation="$repo/src/cf4_phasec_same_gpu_repeatability_diagnostic.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_posterior_v1
readonly device_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_posterior_v1_devices
readonly source_program_sha=19962db42b8476166080309bf8277aade52b86753d66ee967f3d3e8a5c949e7a
readonly device_program_sha=46aba82ea404d13934abe5a1115dca01167a0ffaca322bf7a0e8ac4cb48ee911
readonly device_implementation_sha=79a5080bfb667f5b6294ae741f4246fc516a9b087d498aa4643cd4a7a8d0a8db

: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${DEVICE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$DEVICE_ROOT" == "$device_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == cf4_z0_posterior_v1 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" == syn05 || "$host_name" == syn07 ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" && "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" && "$head_commit" == "$upstream_commit" ]]
[[ "$(sha256sum "$repo/config/cf4_phasec_sampler_mechanics_pilot_v6.json" | awk '{print $1}')" == "$source_program_sha" ]]
[[ "$(sha256sum "$device_program" | awk '{print $1}')" == "$device_program_sha" ]]
[[ "$(sha256sum "$device_implementation" | awk '{print $1}')" == "$device_implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 && "${SLURM_MEM_PER_NODE:-0}" -eq 10240 ]]
command -v nvidia-smi >/dev/null

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

mkdir -p -m 700 "$OUTPUT_ROOT" "$DEVICE_ROOT"
device_task_root=$(printf '%s/task_%02d' "$DEVICE_ROOT" "$SLURM_ARRAY_TASK_ID")
mkdir -p -m 700 "$device_task_root"
"$python" -I -P "$device_implementation" capture-device \
  --program "$device_program" --output-root "$device_task_root"
[[ "$(jq -r '.host' "$device_task_root/device.json")" == "$host_name" ]]
gpu_inventory=$(jq -r '.nvidia_smi_inventory | if length == 1 then .[0] else empty end' "$device_task_root/device.json")
[[ -n "$gpu_inventory" && "$gpu_inventory" != *"GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6"* ]]

"$python" -I -P "$implementation" run \
  --program "$program" --output-root "$OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" --implementation-commit "$EXPECTED_COMMIT" \
  --device-record "$device_task_root/device.json"
