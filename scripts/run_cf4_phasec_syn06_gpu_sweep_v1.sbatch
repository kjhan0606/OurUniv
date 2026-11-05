#!/usr/bin/env bash
#SBATCH --job-name=cf4-syn06-gpusweep1
#SBATCH --partition=a40
#SBATCH --nodelist=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=18432M
#SBATCH --gres=gpu:2
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
readonly program="$repo/config/cf4_phasec_syn06_gpu_sweep_v1.json"
readonly worker="$repo/scripts/run_cf4_phasec_syn06_gpu_sweep_worker_v1.sh"
readonly gate_implementation="$repo/src/cf4_phasec_scan_generator_gate_v2.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/phasec_syn06_gpu_sweep_v1
readonly program_sha=10305f22b40a88370d34bc2c5d1c3c5ced8386e3f0db3262ed01f5bc758fba2b
readonly worker_sha=4591f3519b94ee98064bf9e57143359bc8916f4c473a92671d7e5ac22c01b5b5

: "${SLURM_JOB_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == syn06_gpu_sweep ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
[[ "$(hostname -s)" == syn06 ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$worker" | awk '{print $1}')" == "$worker_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 16 && "${SLURM_MEM_PER_NODE:-0}" -eq 18432 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" -eq 2 ]]
[[ ! -e "$OUTPUT_ROOT" ]]

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

mkdir -m 700 "$OUTPUT_ROOT"
pids=()
for slot in 0 1; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=9216M --gres=gpu:1 \
    --output="$OUTPUT_ROOT/slot_${slot}.out" \
    --error="$OUTPUT_ROOT/slot_${slot}.err" \
    "$worker" "$slot" "$OUTPUT_ROOT" "$EXPECTED_COMMIT" &
  pids+=("$!")
done

worker_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_status=1
done
[[ "$worker_status" -eq 0 ]]

uuid_0=$(jq -r '.nvidia_smi_inventory | if length == 1 then .[0] else empty end' "$OUTPUT_ROOT/slot_0/device.json")
uuid_1=$(jq -r '.nvidia_smi_inventory | if length == 1 then .[0] else empty end' "$OUTPUT_ROOT/slot_1/device.json")
[[ -n "$uuid_0" && -n "$uuid_1" && "$uuid_0" != "$uuid_1" ]]

for slot in 0 1; do
  for task_index in 1 6 2; do
    seed=$((2026083000 + task_index))
    task_dir=$(printf '%s/slot_%d/seed_%02d_%d' "$OUTPUT_ROOT" "$slot" "$task_index" "$seed")
    "$python" -I -P "$gate_implementation" validate-task --directory "$task_dir"
  done
done
