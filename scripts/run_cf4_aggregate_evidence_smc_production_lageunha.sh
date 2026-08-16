#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly source_commit=6630b6b04ab02e513d47f1667617384894eb349f
readonly capability_commit=22587e47232782feb4c08768d8f64d853d76e62b
readonly branch_ref=refs/heads/agent/freeze-zoom-pipeline
readonly tracking_ref=refs/remotes/origin/agent/freeze-zoom-pipeline
readonly program="$repo/config/cf4_aggregate_evidence_smc_production_program.json"
readonly expected_program_sha=74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5
readonly implementation="$repo/src/cf4_aggregate_evidence_smc_execution.py"
readonly execution_tests="$repo/tests/test_cf4_aggregate_evidence_smc_execution.py"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1_run
readonly result="$data/result.json"
readonly manifest="$data/manifest.json"
readonly log="$state/run.log"
readonly environment="$state/environment.txt"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly lock="$state/.runner.lock"

finish() {
    local rc=$? ended_at marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ "${validated_complete:-false}" == true \
          && -s "$result" && -s "$manifest" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        set -o noclobber
        {
            printf 'status=complete\nscience_status=%s\noutcome_kind=%s\n' \
                "$science_status" "$outcome_kind"
            printf 'failure_class=%s\nstarted_at=%s\nended_at=%s\n' \
                "$failure_class" "$started_at" "$ended_at"
            printf 'host=%s\ngit_commit=%s\norigin_commit=%s\nremote_commit=%s\n' \
                "$host" "$commit" "$origin_commit" "$remote_commit"
            printf 'source_commit=%s\ncapability_commit=%s\n' \
                "$source_commit" "$capability_commit"
            printf 'program_sha256=%s\nimplementation_sha256=%s\n' \
                "$program_sha" "$implementation_sha"
            printf 'runner_sha256=%s\nexecution_tests_sha256=%s\n' \
                "$runner_sha" "$execution_tests_sha"
            printf 'manifest=%s\nmanifest_sha256=%s\nresult_sha256=%s\n' \
                "$manifest" "$(sha256sum "$manifest" | awk '{print $1}')" \
                "$(sha256sum "$result" | awk '{print $1}')"
            printf 'environment_sha256=%s\n' \
                "$(sha256sum "$environment" | awk '{print $1}')"
            printf 'automatic_retry_scale_retune_or_follow_on=false\n'
        } >"$marker_tmp"
        set +o noclobber
        mv "$marker_tmp" "$complete"
    else
        marker_tmp="$state/.FAILED.$$"
        set -o noclobber
        {
            printf 'status=failed\nexit_code=%s\n' "$rc"
            printf 'started_at=%s\nended_at=%s\nhost=%s\n' \
                "${started_at:-not_reserved}" "$ended_at" "${host:-unknown}"
            printf 'git_commit=%s\nlog=%s\n' "${commit:-unknown}" "$log"
            printf 'failure_class=invalid_execution_or_postcheck_failure\n'
            printf 'automatic_retry_scale_retune_or_follow_on=false\n'
        } >"$marker_tmp"
        set +o noclobber
        mv "$marker_tmp" "$failed"
    fi
    rm -f "$running"
}

host=$(hostname); readonly host
host_short=${host%%.*}; readonly host_short
if [[ "${host_short,,}" != "$expected_host" ]]; then
    echo "host gate failed: expected $expected_host, found $host" >&2; exit 69
fi
if [[ -e "$state" || -e "$data" ]]; then
    echo "refusing to reuse production SMC state or data" >&2; exit 73
fi
if [[ ! -x "$python" || ! -f "$program" || ! -f "$implementation" \
      || ! -f "$execution_tests" || ! -f "$runner" ]]; then
    echo "missing production SMC environment, program, source, runner, or tests" >&2
    exit 66
fi

commit=$(git -C "$repo" rev-parse HEAD); readonly commit
local_commit=$(git -C "$repo" rev-parse "$branch_ref"); readonly local_commit
origin_commit=$(git -C "$repo" rev-parse "$tracking_ref"); readonly origin_commit
remote_commit=$(git -C "$repo" ls-remote --heads origin "$branch_ref" | awk '{print $1}')
readonly remote_commit
if [[ "$commit" != "$local_commit" || "$commit" != "$origin_commit" \
      || "$commit" != "$remote_commit" ]]; then
    echo "HEAD, local branch, origin tracking ref, and remote ref are not identical" >&2
    exit 65
fi
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" "$commit" \
      || ! git -C "$repo" merge-base --is-ancestor "$capability_commit" "$commit"; then
    echo "audited production lineage is not an ancestor of HEAD" >&2; exit 65
fi

readonly -a science_paths=(
    config/cf4_aggregate_evidence_smc_production_program.json
    src/cf4_aggregate_evidence_smc_execution.py
    tests/test_cf4_aggregate_evidence_smc_execution.py
    scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh
    scripts/launch_cf4_aggregate_evidence_smc_production_lageunha.sh
    scripts/status_cf4_aggregate_evidence_smc_production.sh
    tests/test_cf4_aggregate_evidence_smc_production_runner.py
    config/cf4_aggregate_evidence_smc_capability_implementation_result_record.json
    config/cf4_aggregate_evidence_smc_production_capability_design.json
    src/cf4_aggregate_evidence_parallel_oracle.py
    src/cf4_aggregate_evidence_smc_capability.py
    config/cf4_aggregate_evidence_annealed_smc_design.json
    src/cf4_aggregate_evidence_smc.py
    src/cf4_aggregate_evidence_oracle.py
    src/cf4_aggregate_evidence_smc_validation.py
    src/cf4_aggregate_evidence_smc_production.py
)
env REPO="$repo" GIT_COMMIT="$commit" PYTHONPATH="$repo/src" "$python" - \
    "${science_paths[@]}" <<'PY'
import os
from pathlib import Path
import subprocess
import sys

repo = Path(os.environ["REPO"])
commit = os.environ["GIT_COMMIT"]
paths = sys.argv[1:]
for path in paths:
    subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", path],
        check=True,
        stdout=subprocess.DEVNULL,
    )
subprocess.run(
    ["git", "-C", str(repo), "diff", "--quiet", commit, "--", *paths],
    check=True,
)
PY

program_sha=$(sha256sum "$program" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
execution_tests_sha=$(sha256sum "$execution_tests" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
readonly program_sha implementation_sha execution_tests_sha runner_sha
if [[ "$program_sha" != "$expected_program_sha" ]]; then
    echo "production program hash mismatch" >&2; exit 65
fi

env PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_execution import load_canonical_program

program = load_canonical_program(verify_file_hashes=True)
if program["authorization"].get("production_execution_authorized") is not True:
    raise SystemExit("production SMC execution remains unauthorized")
PY

available_kib=$(df -Pk /gpfs/kjhan/CF4/recon/linear_cr | awk 'NR == 2 {print $4}')
memory_kib=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)
readonly available_kib memory_kib
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < 41943040 )); then
    echo "production SMC requires at least 40 GiB free" >&2; exit 70
fi
if [[ ! "$memory_kib" =~ ^[0-9]+$ ]] || (( memory_kib < 67108864 )); then
    echo "production SMC requires at least 64 GiB MemAvailable" >&2; exit 70
fi
if [[ -e "$state" || -e "$data" ]]; then
    echo "production SMC state or data appeared during read-only preflight" >&2
    exit 75
fi

started_at=$(date --iso-8601=seconds); readonly started_at
mkdir "$state"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$data"
exec 9>"$lock"
if ! flock -n 9; then
    echo "failed to acquire the exclusive production SMC reservation" >&2; exit 75
fi

set -o noclobber
: >"$log"
: >"$environment"
set +o noclobber
exec >>"$log" 2>&1

running_tmp="$state/.RUNNING.$$"
set -o noclobber
{
    printf 'status=running\nstage=aggregate_evidence_smc_production\n'
    printf 'pid=%s\nstarted_at=%s\nhost=%s\ngit_commit=%s\n' \
        "$$" "$started_at" "$host" "$commit"
    printf 'origin_commit=%s\nremote_commit=%s\nsource_commit=%s\ncapability_commit=%s\n' \
        "$origin_commit" "$remote_commit" "$source_commit" "$capability_commit"
    printf 'worker_processes=8\nthreads_per_worker=1\nreplicates_sequential=true\n'
    printf 'program_sha256=%s\nrunner_sha256=%s\nlog=%s\n' \
        "$program_sha" "$runner_sha" "$log"
} >"$running_tmp"
set +o noclobber
mv "$running_tmp" "$running"

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\norigin_commit=%s\nremote_commit=%s\nCUDA_VISIBLE_DEVICES=\n' \
        "$host" "$commit" "$origin_commit" "$remote_commit"
    printf 'worker_processes=8\nthreads_per_worker=1\nreplicates_sequential=true\n'
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_aggregate_evidence_smc_execution.py \
    --program "$program"
test -s "$result"
test -s "$manifest"

if [[ "$(git -C "$repo" rev-parse HEAD)" != "$commit" \
      || "$(git -C "$repo" rev-parse "$tracking_ref")" != "$origin_commit" \
      || "$(git -C "$repo" ls-remote --heads origin "$branch_ref" | awk '{print $1}')" != "$remote_commit" \
      || "$(sha256sum "$program" | awk '{print $1}')" != "$program_sha" \
      || "$(sha256sum "$implementation" | awk '{print $1}')" != "$implementation_sha" \
      || "$(sha256sum "$execution_tests" | awk '{print $1}')" != "$execution_tests_sha" \
      || "$(sha256sum "$runner" | awk '{print $1}')" != "$runner_sha" ]]; then
    echo "production SMC HEAD, origin, or source hash changed during execution" >&2
    exit 65
fi
env REPO="$repo" GIT_COMMIT="$commit" "$python" - \
    "${science_paths[@]}" <<'PY'
import os
from pathlib import Path
import subprocess
import sys

subprocess.run(
    ["git", "-C", os.environ["REPO"], "diff", "--quiet",
     os.environ["GIT_COMMIT"], "--", *sys.argv[1:]],
    check=True,
)
PY
env PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_execution import load_canonical_program
load_canonical_program(verify_file_hashes=True)
PY

postcheck=$(env PYTHONPATH="$repo/src" "$python" - "$data" <<'PY'
from pathlib import Path
import sys

from cf4_aggregate_evidence_smc_execution import validate_published_bundle

value = validate_published_bundle(Path(sys.argv[1]))
failure = value["failure_class"] if value["failure_class"] is not None else "none"
print(value["status"], value["outcome_kind"], failure, value["valid_scientific_complete"])
PY
)
read -r science_status outcome_kind failure_class validated_complete <<<"$postcheck"
if [[ "$validated_complete" != true \
      || "$science_status" != complete_pass_production_smc \
         && "$science_status" != complete_scientific_fail_production_smc ]]; then
    echo "production SMC inline postcheck did not produce a valid scientific COMPLETE" >&2
    exit 65
fi
readonly science_status outcome_kind failure_class validated_complete
printf '[runner] status=%s outcome=%s failure=%s\n' \
    "$science_status" "$outcome_kind" "$failure_class"
