#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly authorization_design_base_commit=d3213fa8fa2effe82dc6874911d21132dc088b4b
readonly runner_implementation_commit=375438fa6dc911059da57e46be95183ee45f1837
readonly branch_ref=refs/heads/agent/freeze-zoom-pipeline
readonly tracking_ref=refs/remotes/origin/agent/freeze-zoom-pipeline
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v2.json"
readonly expected_program_sha=b028cf09045a46061f899dc1a8b0212a1ebe586e0b5d27f3f159a47f48d3a244
readonly grant="$repo/config/cf4_aggregate_evidence_smc_execution_grant_v2.json"
readonly release=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v2_release.json
readonly implementation="$repo/src/cf4_aggregate_evidence_smc_execution_authorized_v2.py"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_authorized_v2_lageunha.sh"
readonly status_checker="$repo/scripts/status_cf4_aggregate_evidence_smc_production.sh"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v2
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v2_run
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
          && -s "$result" && -s "$manifest" && -f "$release" \
          && "$(sha256sum "$release" | awk '{print $1}')" == "$release_sha" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        set -o noclobber
        {
            printf 'status=complete\nscience_status=%s\noutcome_kind=%s\n' \
                "$science_status" "$outcome_kind"
            printf 'failure_class=%s\nstarted_at=%s\nended_at=%s\n' \
                "$failure_class" "$started_at" "$ended_at"
            printf 'host=%s\ngit_commit=%s\norigin_commit=%s\nremote_commit=%s\n' \
                "$host" "$commit" "$origin_commit" "$remote_commit"
            printf 'authorization_program_sha256=%s\ngrant_sha256=%s\n' \
                "$program_sha" "$grant_sha"
            printf 'external_lineage_release=%s\nexternal_lineage_release_sha256=%s\n' \
                "$release" "$release_sha"
            printf 'authorization_source_sha256=%s\nrunner_sha256=%s\n' \
                "$implementation_sha" "$runner_sha"
            printf 'manifest=%s\nmanifest_sha256=%s\nresult_sha256=%s\n' \
                "$manifest" "$(sha256sum "$manifest" | awk '{print $1}')" \
                "$(sha256sum "$result" | awk '{print $1}')"
            printf 'environment_sha256=%s\n' \
                "$(sha256sum "$environment" | awk '{print $1}')"
            printf 'automatic_retry_retune_scale_up_or_follow_on=false\n'
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
            printf 'automatic_retry_retune_scale_up_or_follow_on=false\n'
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
      || ! -f "$runner" || ! -x "$status_checker" ]]; then
    echo "missing authorization program, source, runner, status, or environment" >&2
    exit 66
fi

commit=$(git -C "$repo" rev-parse HEAD); readonly commit
local_commit=$(git -C "$repo" rev-parse "$branch_ref"); readonly local_commit
origin_commit=$(git -C "$repo" rev-parse "$tracking_ref"); readonly origin_commit
remote_commit=$(git -C "$repo" ls-remote --heads origin "$branch_ref" | awk '{print $1}')
readonly remote_commit
if [[ "$commit" != "$local_commit" || "$commit" != "$origin_commit" \
      || "$commit" != "$remote_commit" ]]; then
    echo "HEAD, local branch, origin tracking ref, and remote ref differ" >&2; exit 65
fi
if ! git -C "$repo" merge-base --is-ancestor \
        "$authorization_design_base_commit" "$commit" \
      || ! git -C "$repo" merge-base --is-ancestor \
        "$runner_implementation_commit" "$commit"; then
    echo "audited authorization or runner lineage is not an ancestor" >&2; exit 65
fi

readonly -a science_paths=(
    config/cf4_aggregate_evidence_smc_execution_authorization_design_v2.json
    config/cf4_aggregate_evidence_smc_execution_authorization_program_v2.json
    src/cf4_aggregate_evidence_smc_execution_authorized_v2.py
    tests/test_cf4_aggregate_evidence_smc_execution_authorized_v2.py
    scripts/run_cf4_aggregate_evidence_smc_authorized_v2_lageunha.sh
    scripts/launch_cf4_aggregate_evidence_smc_authorized_v2_lageunha.sh
    tests/test_cf4_aggregate_evidence_smc_authorized_v2_runner.py
    config/cf4_aggregate_evidence_smc_runner_implementation_result_record.json
    config/cf4_aggregate_evidence_smc_production_program.json
    src/cf4_aggregate_evidence_smc_execution.py
    scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh
    scripts/launch_cf4_aggregate_evidence_smc_production_lageunha.sh
    scripts/status_cf4_aggregate_evidence_smc_production.sh
)
env REPO="$repo" GIT_COMMIT="$commit" "$python" - \
    "${science_paths[@]}" <<'PY'
import os
import subprocess
import sys

repo = os.environ["REPO"]
commit = os.environ["GIT_COMMIT"]
for path in sys.argv[1:]:
    subprocess.run(
        ["git", "-C", repo, "ls-files", "--error-unmatch", path],
        check=True,
        stdout=subprocess.DEVNULL,
    )
subprocess.run(
    ["git", "-C", repo, "diff", "--quiet", commit, "--", *sys.argv[1:]],
    check=True,
)
PY

program_sha=$(sha256sum "$program" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
readonly program_sha implementation_sha runner_sha
if [[ "$program_sha" != "$expected_program_sha" ]]; then
    echo "authorization program hash mismatch" >&2; exit 65
fi

# This read-only gate is deliberately before disk checks and every reservation.
# With the repository shipped by this change it always refuses: the program is
# execution-false and the separately audited canonical grant does not exist.
env PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_execution_authorized_v2 import (
    load_canonical_authorization_program,
    require_execution_authorization,
)

program = load_canonical_authorization_program(verify_file_hashes=False)
require_execution_authorization(program)
load_canonical_authorization_program(verify_file_hashes=True)
PY
grant_sha=$(sha256sum "$grant" | awk '{print $1}'); readonly grant_sha
release_sha=$(sha256sum "$release" | awk '{print $1}'); readonly release_sha

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
    echo "production SMC state or data appeared during preflight" >&2; exit 75
fi

started_at=$(date --iso-8601=seconds); readonly started_at
mkdir "$state"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$data"
exec 9>"$lock"
if ! flock -n 9; then
    echo "failed to acquire exclusive production SMC reservation" >&2; exit 75
fi
set -o noclobber
: >"$log"
: >"$environment"
set +o noclobber
exec >>"$log" 2>&1

running_tmp="$state/.RUNNING.$$"
set -o noclobber
{
    printf 'status=running\nstage=aggregate_evidence_smc_authorized_v2\n'
    printf 'pid=%s\nstarted_at=%s\nhost=%s\ngit_commit=%s\n' \
        "$$" "$started_at" "$host" "$commit"
    printf 'origin_commit=%s\nremote_commit=%s\n' "$origin_commit" "$remote_commit"
    printf 'worker_processes=8\nthreads_per_worker=1\nreplicates_sequential=true\n'
    printf 'authorization_program_sha256=%s\ngrant_sha256=%s\nlog=%s\n' \
        "$program_sha" "$grant_sha" "$log"
    printf 'external_lineage_release=%s\nexternal_lineage_release_sha256=%s\n' \
        "$release" "$release_sha"
} >"$running_tmp"
set +o noclobber
mv "$running_tmp" "$running"

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\norigin_commit=%s\nremote_commit=%s\n' \
        "$host" "$commit" "$origin_commit" "$remote_commit"
    printf 'external_lineage_release=%s\nexternal_lineage_release_sha256=%s\n' \
        "$release" "$release_sha"
    printf 'worker_processes=8\nthreads_per_worker=1\nreplicates_sequential=true\n'
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_aggregate_evidence_smc_execution_authorized_v2.py \
    --program "$program"
test -s "$result"
test -s "$manifest"

if [[ ! -f "$release" \
      || "$(sha256sum "$release" | awk '{print $1}')" != "$release_sha" \
      || "$(git -C "$repo" rev-parse HEAD)" != "$commit" \
      || "$(git -C "$repo" rev-parse "$tracking_ref")" != "$origin_commit" \
      || "$(git -C "$repo" ls-remote --heads origin "$branch_ref" | awk '{print $1}')" != "$remote_commit" \
      || "$(sha256sum "$program" | awk '{print $1}')" != "$program_sha" \
      || "$(sha256sum "$grant" | awk '{print $1}')" != "$grant_sha" \
      || "$(sha256sum "$implementation" | awk '{print $1}')" != "$implementation_sha" \
      || "$(sha256sum "$runner" | awk '{print $1}')" != "$runner_sha" ]]; then
    echo "authorization lineage or source changed during execution" >&2; exit 65
fi

env PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_execution_authorized_v2 import (
    load_canonical_authorization_program,
    require_execution_authorization,
)

program = load_canonical_authorization_program(verify_file_hashes=True)
require_execution_authorization(program)
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
# Python's string form of bool is "True"/"False"; normalize the wire value
# before the shell completion gate and EXIT trap consume it.
validated_complete=${validated_complete,,}
if [[ "$validated_complete" != true \
      || "$science_status" != complete_pass_production_smc \
         && "$science_status" != complete_scientific_fail_production_smc ]]; then
    echo "authorized SMC inline postcheck was not valid scientific COMPLETE" >&2
    exit 65
fi
readonly science_status outcome_kind failure_class validated_complete
printf '[authorized-v2-runner] status=%s outcome=%s failure=%s\n' \
    "$science_status" "$outcome_kind" "$failure_class"
