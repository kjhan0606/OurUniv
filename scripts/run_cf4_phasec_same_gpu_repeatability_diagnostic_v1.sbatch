#!/usr/bin/env bash
#SBATCH --job-name=cf4-samegpu-rep1
#SBATCH --partition=a40,a100,h200,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9216M
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%j.err
#SBATCH --open-mode=truncate
set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_phasec_same_gpu_repeatability_diagnostic_v1.json"
readonly implementation="$repo/src/cf4_phasec_same_gpu_repeatability_diagnostic.py"
readonly gate_program="$repo/config/cf4_phasec_scan_generator_gate_v2.json"
readonly gate_implementation="$repo/src/cf4_phasec_scan_generator_gate_v2.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/phasec_same_gpu_repeatability_v1
readonly aggregate_output=/gpfs/kjhan/CF4/z0_density/phasec_same_gpu_repeatability_v1_aggregate
readonly program_sha=46aba82ea404d13934abe5a1115dca01167a0ffaca322bf7a0e8ac4cb48ee911
readonly implementation_sha=79a5080bfb667f5b6294ae741f4246fc516a9b087d498aa4643cd4a7a8d0a8db
readonly gate_program_sha=07b1c0434b954a364f66d78ab6e6b8c63cfe5ba609909ad16b8b2fd897980cf7
readonly gate_implementation_sha=5552198373b2c1507d981c467de9b9a469e053a8af189e51aea20495bf3c8f4b
readonly gate_implementation_commit=b4f33d2a1f408c07ba746a8765d78096a6028e5c

: "${SLURM_JOB_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${AGGREGATE_OUTPUT:?}"
: "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$AGGREGATE_OUTPUT" == "$aggregate_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == same_gpu_repeatability ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* ]]
[[ "$host_name" != syn101 && "$host_name" != syn101.* ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ "$(sha256sum "$gate_program" | awk '{print $1}')" == "$gate_program_sha" ]]
[[ "$(sha256sum "$gate_implementation" | awk '{print $1}')" == "$gate_implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 && "${SLURM_MEM_PER_NODE:-0}" -eq 9216 ]]
[[ ! -e "$OUTPUT_ROOT" && ! -e "$AGGREGATE_OUTPUT" ]]
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

mkdir -m 700 "$OUTPUT_ROOT"
"$python" -I -P "$implementation" capture-device \
  --program "$program" \
  --output-root "$OUTPUT_ROOT"

for repeat in 0 1; do
  for task_index in 1 6 2 7; do
    "$python" -I -P "$gate_implementation" run \
      --program "$gate_program" \
      --output-root "$OUTPUT_ROOT/repeat_$repeat" \
      --task-index "$task_index" \
      --implementation-commit "$gate_implementation_commit"
  done
done

"$python" -I -P "$implementation" aggregate \
  --program "$program" \
  --output-root "$OUTPUT_ROOT" \
  --aggregate-output "$AGGREGATE_OUTPUT" \
  --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$implementation" validate-aggregate --directory "$AGGREGATE_OUTPUT"
