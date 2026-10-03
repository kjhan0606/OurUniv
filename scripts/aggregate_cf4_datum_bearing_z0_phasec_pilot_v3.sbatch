#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-phasec-v3-agg
#SBATCH --partition=a10,a40,a100,h200,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
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
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_execution_amendment_v3.json"
readonly implementation="$repo/src/cf4_datum_bearing_z0_phasec_pilot.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v3
readonly aggregate_output=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v3_aggregate
readonly program_sha=f2c52c834e133ec4e8702275f63214d37fdeb0ab1f53a83e2b6e6a99d83bfb50
readonly implementation_sha=657f3353e9105ba7a10babb82c6821ae0c8def2a2e1bb12047fffd1bf7a3c1de

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be explicitly exported}"
: "${AGGREGATE_OUTPUT:?AGGREGATE_OUTPUT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"
[[ "$OUTPUT_ROOT" == "$output_root" ]]
[[ "$AGGREGATE_OUTPUT" == "$aggregate_output" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == aggregate ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ ! -e "$aggregate_output" ]]
[[ ! -e "/gpfs/kjhan/CF4/z0_density/.datum_bearing_phasec_pilot_v3_aggregate.staging" ]]
readonly cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 1024 ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0
"$python" -I -P "$implementation" aggregate \
  --program "$program" --output-root "$OUTPUT_ROOT" \
  --aggregate-output "$AGGREGATE_OUTPUT" --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$implementation" validate-aggregate --directory "$AGGREGATE_OUTPUT"
