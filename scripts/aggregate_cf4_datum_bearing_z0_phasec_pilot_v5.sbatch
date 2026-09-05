#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-phasec-v5-agg
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
readonly repo=/home/kjhan/BACKUP/CF4 python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_execution_amendment_v5.json" implementation="$repo/src/cf4_datum_bearing_z0_phasec_pilot.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v5 aggregate_output=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v5_aggregate
readonly program_sha=4bafc8a51fc341979a5245d4b6f5341c6aaf36dec0e7135c2dfc1acee65dc556 implementation_sha=79225219fb170cbe19d5be617f070444b2edd351cecabb8941d7558e77744f11
: "${SLURM_JOB_ID:?}" "${SLURM_SUBMIT_HOST:?}" "${OUTPUT_ROOT:?}" "${AGGREGATE_OUTPUT:?}" "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}" "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" && "$AGGREGATE_OUTPUT" == "$aggregate_output" && "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == aggregate ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ ! -e "$aggregate_output" && ! -e "/gpfs/kjhan/CF4/z0_density/.datum_bearing_phasec_pilot_v5_aggregate.staging" ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD) upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" && "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" && "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" && "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 1 && "${SLURM_MEM_PER_NODE:-0}" -eq 1024 ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0
"$python" -I -P "$implementation" aggregate --program "$program" --output-root "$OUTPUT_ROOT" --aggregate-output "$AGGREGATE_OUTPUT" --implementation-commit "$EXPECTED_COMMIT"
"$python" -I -P "$implementation" validate-aggregate --directory "$AGGREGATE_OUTPUT"
