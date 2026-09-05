#!/usr/bin/env bash
#SBATCH --job-name=cf4-phasec-rg1-agg
#SBATCH --partition=a40,a100,h200,a100_pcie
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
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_redesign_v1.json"
readonly implementation="$repo/src/cf4_phasec_redesign_generator_gate.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_redesign_v1_generator
readonly aggregate_output=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_redesign_v1_generator_aggregate
readonly program_sha=f5817e71c8038e64825516f01ddfd8e61a077ef664cf5b73acbc4a4bd1c14a76
readonly implementation_sha=1751948d93dd18d25f0cdb1b8139373809bd8380baf62d558a95d2daae124afc

: "${SLURM_JOB_ID:?}" "${SLURM_SUBMIT_HOST:?}" "${OUTPUT_ROOT:?}" "${AGGREGATE_OUTPUT:?}"
: "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}" "${SUBMISSION_CONTROLLER:?}"
: "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$AGGREGATE_OUTPUT" == "$aggregate_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == generator_aggregate ]]
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
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 && "${SLURM_MEM_PER_NODE:-0}" -eq 1024 ]]

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
