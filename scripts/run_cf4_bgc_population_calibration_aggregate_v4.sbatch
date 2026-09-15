#!/usr/bin/env bash
#SBATCH --job-name=cf4-bgc-pop-agg-v4
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4096M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1/logs/aggregate-v4-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1/logs/aggregate-v4-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
program=config/cf4_bgc_population_calibration_program_v1.json
correction=config/cf4_bgc_population_calibration_aggregate_correction_v4.json
source_file=src/cf4_population_calibration_aggregate_v2.py
base_source=src/cf4_population_calibration.py
v3_correction=config/cf4_bgc_population_calibration_aggregate_correction_v3.json
failed_v3_runner=scripts/run_cf4_bgc_population_calibration_aggregate_v3.sbatch
output_root=/gpfs/kjhan/CF4/kf_design/bgc_population_calibration_v1
members_root=${output_root}/members
output=${output_root}/aggregate
program_sha=38ff44a066a55048f9d35ef93c64cce2f98702b4a785099087ffc97a698a00f0
correction_sha=17280db23b952293031e643a1f18e3bc14246b9827027594aa07c8369495acf0
source_sha=371016743b9c0609c15a400f30560cc288ac694301f59f482e3fa0cb132a9969
base_source_sha=fe3b1dd38258b2ed28fd27a551287e14e797232f8f89da83ae1cc59c3309b2af
v3_correction_sha=a8a1054d7e9c181afdcb3677cdf86b41375f1bf4121b3a4afb9340a8d6db4f57
failed_v3_runner_sha=9a39340997696ba17b0ed9a9e72fa119600895f22d7b7544fbaef5837fd81768
expected_comment=cf4_bgc_pop_v4_accounted_328686_failed_328695_328769_328782

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${MEMBER_IMPLEMENTATION_COMMIT:?MEMBER_IMPLEMENTATION_COMMIT must be explicitly exported}"
: "${MEMBER_ARRAY_JOB_ID:?MEMBER_ARRAY_JOB_ID must be explicitly exported}"
: "${FAILED_V1_JOB_ID:?FAILED_V1_JOB_ID must be explicitly exported}"
: "${FAILED_V2_JOB_ID:?FAILED_V2_JOB_ID must be explicitly exported}"
: "${FAILED_V3_JOB_ID:?FAILED_V3_JOB_ID must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"

[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit_single_v4_aggregate_correction ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$MEMBER_IMPLEMENTATION_COMMIT" == 342bb7c77ac60801a303cb81311a30b3506c8f1d ]]
[[ "$MEMBER_ARRAY_JOB_ID" == 328686 ]]
[[ "$FAILED_V1_JOB_ID" == 328695 ]]
[[ "$FAILED_V2_JOB_ID" == 328769 ]]
[[ "$FAILED_V3_JOB_ID" == 328782 ]]
[[ -d "$output_root" ]]
[[ -d "$members_root" ]]
[[ ! -e "$output" ]]
[[ ! -e "${output_root}/.aggregate.staging" ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]

cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]
job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$job_record" == *"Comment=${expected_comment}"* ]]
declare -A member_state
while IFS='|' read -r job_id state exit_code; do
  [[ -n "$job_id" ]]
  [[ -z "${member_state[$job_id]+present}" ]]
  member_state[$job_id]="${state}|${exit_code}"
done < <(sacct -j "$MEMBER_ARRAY_JOB_ID" --starttime 2026-08-31 -X -n -P -o JobID,State,ExitCode)
[[ "${#member_state[@]}" -eq 64 ]]
for index in $(seq 0 63); do
  [[ "${member_state[${MEMBER_ARRAY_JOB_ID}_${index}]-missing}" == "COMPLETED|0:0" ]]
done

head_commit=$(git -C "$repo" rev-parse HEAD)
upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$correction" | awk '{print $1}')" == "$correction_sha" ]]
[[ "$(sha256sum "$source_file" | awk '{print $1}')" == "$source_sha" ]]
[[ "$(sha256sum "$base_source" | awk '{print $1}')" == "$base_source_sha" ]]
[[ "$(sha256sum "$v3_correction" | awk '{print $1}')" == "$v3_correction_sha" ]]
[[ "$(sha256sum "$failed_v3_runner" | awk '{print $1}')" == "$failed_v3_runner_sha" ]]

"$python" -I -P - "$program" "$correction" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

program = json.loads(Path(sys.argv[1]).read_text())
correction = json.loads(Path(sys.argv[2]).read_text())
assert sys.version.split()[0] == program["environment"]["python_version"]
for package in ("numpy", "jax", "jaxlib", "pmwd", "scipy"):
    assert metadata.version(package) == program["environment"][package]
assert [row["Slurm_job_id"] for row in correction["preserved_failed_jobs"]] == [328695, 328769, 328782]
assert correction["root_cause_and_fix"]["density_bin_11_mode_count"] == 3
assert correction["root_cause_and_fix"]["theta_bin_11_mode_count"] == 0
assert correction["root_cause_and_fix"]["truth_metric_or_candidate_value_used_to_choose_fix"] is False
assert correction["authorization"]["single_v4_aggregate_correction_submission"] is True
assert correction["authorization"]["member_rerun"] is False
assert correction["authorization"]["automatic_retry_after_v4"] is False
execution = correction["execution"]
assert execution["maximum_v4_submissions"] == 1
assert execution["requested_memory_MiB"] >= 1.2 * execution["expected_peak_memory_MiB"]
PY

[[ "$(readlink -f "$python")" == "$python" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 4096 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE}" == 0 ]]

export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P "$source_file" aggregate \
  --program "$program" \
  --members-root "$members_root" \
  --output "$output" \
  --member-implementation-commit "$MEMBER_IMPLEMENTATION_COMMIT" \
  --aggregation-runtime-commit "$EXPECTED_COMMIT"

"$python" -I -P "$source_file" validate \
  --directory "$output"
