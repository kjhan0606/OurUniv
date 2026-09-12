#!/usr/bin/env bash
#SBATCH --job-name=cf4-scan-gen3
#SBATCH --partition=a40
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9216M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
readonly rerun_program="$repo/config/cf4_phasec_scan_generator_gate_v3_hardware_quarantine.json"
readonly device_program="$repo/config/cf4_phasec_same_gpu_repeatability_diagnostic_v1.json"
readonly device_implementation="$repo/src/cf4_phasec_same_gpu_repeatability_diagnostic.py"
readonly gate_program="$repo/config/cf4_phasec_scan_generator_gate_v2.json"
readonly gate_implementation="$repo/src/cf4_phasec_scan_generator_gate_v2.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/phasec_scan_generator_gate_v3
readonly device_root=/gpfs/kjhan/CF4/z0_density/phasec_scan_generator_gate_v3_devices
readonly rerun_program_sha=c665c94c061be3e768a53f6a4fd882be02cd9cae5ba9b2de632d8216c805f171
readonly device_program_sha=46aba82ea404d13934abe5a1115dca01167a0ffaca322bf7a0e8ac4cb48ee911
readonly device_implementation_sha=79a5080bfb667f5b6294ae741f4246fc516a9b087d498aa4643cd4a7a8d0a8db
readonly gate_program_sha=07b1c0434b954a364f66d78ab6e6b8c63cfe5ba609909ad16b8b2fd897980cf7
readonly gate_implementation_sha=5552198373b2c1507d981c467de9b9a469e053a8af189e51aea20495bf3c8f4b
readonly gate_implementation_commit=b4f33d2a1f408c07ba746a8765d78096a6028e5c
readonly failing_uuid=GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6

: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${DEVICE_ROOT:?}"
: "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$DEVICE_ROOT" == "$device_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == scan_generator_gate_v3 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" == syn05 || "$host_name" == syn07 ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$rerun_program" | awk '{print $1}')" == "$rerun_program_sha" ]]
[[ "$(sha256sum "$device_program" | awk '{print $1}')" == "$device_program_sha" ]]
[[ "$(sha256sum "$device_implementation" | awk '{print $1}')" == "$device_implementation_sha" ]]
[[ "$(sha256sum "$gate_program" | awk '{print $1}')" == "$gate_program_sha" ]]
[[ "$(sha256sum "$gate_implementation" | awk '{print $1}')" == "$gate_implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 && "${SLURM_MEM_PER_NODE:-0}" -eq 9216 ]]

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

device_task_root=$(printf '%s/task_%02d' "$DEVICE_ROOT" "$SLURM_ARRAY_TASK_ID")
[[ ! -e "$device_task_root" ]]
mkdir -p -m 700 "$device_task_root"
"$python" -I -P "$device_implementation" capture-device \
  --program "$device_program" \
  --output-root "$device_task_root"
[[ "$(jq -r '.host' "$device_task_root/device.json")" == "$host_name" ]]
gpu_inventory=$(jq -r '.nvidia_smi_inventory | if length == 1 then .[0] else empty end' "$device_task_root/device.json")
[[ -n "$gpu_inventory" && "$gpu_inventory" != *"$failing_uuid"* ]]

"$python" -I -P "$gate_implementation" run \
  --program "$gate_program" \
  --output-root "$OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --implementation-commit "$gate_implementation_commit"
seed=$((2026083000 + SLURM_ARRAY_TASK_ID))
task_dir=$(printf '%s/seed_%02d_%d' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID" "$seed")
"$python" -I -P "$gate_implementation" validate-task --directory "$task_dir"
