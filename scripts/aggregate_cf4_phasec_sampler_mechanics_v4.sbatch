#!/usr/bin/env bash
#SBATCH --job-name=cf4-sampler-v4-agg
#SBATCH --partition=a40
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.err
#SBATCH --open-mode=truncate
set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_phasec_sampler_mechanics_pilot_v4.json"
readonly implementation="$repo/src/cf4_phasec_sampler_mechanics_pilot_v4.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/phasec_sampler_mechanics_v4
readonly device_root=/gpfs/kjhan/CF4/z0_density/phasec_sampler_mechanics_v4_devices
readonly aggregate_output=/gpfs/kjhan/CF4/z0_density/phasec_sampler_mechanics_v4_aggregate
readonly program_sha=bb5b826529dd45fac6680c9a0b1cd1dd6d4954115ecd9c47ebb92a17e8ed0f96
readonly implementation_sha=be50a35d9090f7c9a9914b8fafeebf633c25f97b25c943215896e9525514d014
readonly failing_uuid=GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6

: "${SLURM_JOB_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${DEVICE_ROOT:?}" "${AGGREGATE_OUTPUT:?}"
: "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$DEVICE_ROOT" == "$device_root" ]]
[[ "$AGGREGATE_OUTPUT" == "$aggregate_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == phasec_sampler_mechanics_aggregate_v4 ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" == syn05 || "$host_name" == syn07 ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 && "${SLURM_MEM_PER_NODE:-0}" -eq 1024 ]]

for task_index in 0 1; do
  device_file=$(printf '%s/task_%02d/device.json' "$DEVICE_ROOT" "$task_index")
  [[ -f "$device_file" ]]
  task_host=$(jq -r '.host' "$device_file")
  [[ "$task_host" == syn05 || "$task_host" == syn07 ]]
  gpu_inventory=$(jq -r '.nvidia_smi_inventory | if length == 1 then .[0] else empty end' "$device_file")
  [[ -n "$gpu_inventory" && "$gpu_inventory" != *"$failing_uuid"* ]]
done

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P "$implementation" aggregate \
  --program "$program" \
  --output-root "$OUTPUT_ROOT" \
  --aggregate-output "$AGGREGATE_OUTPUT" \
  --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$implementation" validate-aggregate --directory "$AGGREGATE_OUTPUT"
