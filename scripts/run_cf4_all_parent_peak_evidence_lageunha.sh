#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly program="$repo/config/cf4_all_parent_peak_evidence_program.json"
readonly expected_program_sha=9280ccab55d32771245df1e8b230bbac4f924d7ead8173afa404b8fb542af77d
readonly implementation="$repo/src/cf4_all_parent_peak_evidence.py"
readonly expected_implementation_sha=b82bc7b035d39662acef37c35673b3b064a323056a752be2e33d440b3e20a64e
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/all_parent_peak_evidence_v1
readonly output="$state/result.json"
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly environment="$state/environment.txt"
readonly lock="$state/.runner.lock"

mkdir -p "$state"
exec 9>"$lock"
if ! flock -n 9; then
    echo "another all-parent evidence runner owns $lock" >&2
    exit 75
fi

host=$(hostname)
readonly host
host_short=${host%%.*}
readonly host_short
if [[ "${host_short,,}" != "$expected_host" ]]; then
    echo "host gate failed: expected $expected_host, found $host" >&2
    exit 69
fi
if [[ -e "$output" || -e "$running" || -e "$complete" || -e "$failed" \
      || -e "$log" || -e "$environment" ]]; then
    echo "refusing to overwrite an all-parent evidence file" >&2
    exit 73
fi
if [[ ! -x "$python" || ! -f "$program" || ! -f "$implementation" ]]; then
    echo "missing Python environment, program, or implementation" >&2
    exit 66
fi

program_sha=$(sha256sum "$program" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
readonly program_sha implementation_sha
if [[ "$program_sha" != "$expected_program_sha" ]]; then
    echo "program hash mismatch: $program_sha" >&2
    exit 65
fi
if [[ "$implementation_sha" != "$expected_implementation_sha" ]]; then
    echo "implementation hash mismatch: $implementation_sha" >&2
    exit 65
fi

readonly -a science_paths=(
    config/cf4_all_parent_peak_evidence_program.json
    config/cf4_independent_parent_architecture_design.json
    config/cf4_peak_evidence_phase_control_v2_result_record.json
    config/p2_lg_z0_forward_importance_v8.json
    src/cf4_all_parent_peak_evidence.py
    src/cf4_peak_evidence_phase_cache.py
    src/cf4_lg_peak_cr.py
)
for path in "${science_paths[@]}"; do
    git -C "$repo" ls-files --error-unmatch "$path" >/dev/null
done
if ! git -C "$repo" diff --quiet HEAD -- "${science_paths[@]}"; then
    echo "science code/config differs from tracked HEAD" >&2
    exit 65
fi

set -o noclobber
: >"$log"
: >"$environment"
set +o noclobber
exec >>"$log" 2>&1

started_at=$(date --iso-8601=seconds)
commit=$(git -C "$repo" rev-parse HEAD)
readonly started_at commit

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

finish() {
    local rc=$? ended_at marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ -s "$output" && -n "${science_status:-}" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\nscience_status=%s\n' "$science_status"
            printf 'feasibility_pass=%s\nconditional_field_bank_authorized=%s\n' \
                "$feasibility_pass" "$conditional_bank_authorized"
            printf 'candidate_generation_authorized=%s\nPM_or_RAMSES_authorized=%s\n' \
                "$candidate_authorized" "$pm_authorized"
            printf 'started_at=%s\nended_at=%s\nhost=%s\n' "$started_at" "$ended_at" "$host"
            printf 'git_commit=%s\nprogram_sha256=%s\nimplementation_sha256=%s\n' \
                "$commit" "$program_sha" "$implementation_sha"
            printf 'output=%s\noutput_sha256=%s\n' "$output" \
                "$(sha256sum "$output" | awk '{print $1}')"
            printf 'environment_sha256=%s\n' \
                "$(sha256sum "$environment" | awk '{print $1}')"
        } >"$marker_tmp"
        mv "$marker_tmp" "$complete"
    else
        marker_tmp="$state/.FAILED.$$"
        {
            printf 'status=failed\nexit_code=%s\nstarted_at=%s\nended_at=%s\n' \
                "$rc" "$started_at" "$ended_at"
            printf 'host=%s\ngit_commit=%s\nlog=%s\n' "$host" "$commit" "$log"
        } >"$marker_tmp"
        mv "$marker_tmp" "$failed"
    fi
    rm -f "$running"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

running_tmp="$state/.RUNNING.$$"
{
    printf 'status=running\npid=%s\nstarted_at=%s\n' "$$" "$started_at"
    printf 'host=%s\ngit_commit=%s\n' "$host" "$commit"
    printf 'implementation_sha256=%s\nprogram_sha256=%s\n' \
        "$implementation_sha" "$program_sha"
    printf 'worker_processes=8\nthreads_per_worker=1\nlog=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s workers=8 threads=1\n' \
    "$started_at" "$host" "$$" "$commit"
printf '[runner] implementation_sha256=%s program_sha256=%s\n' \
    "$implementation_sha" "$program_sha"
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\n' "$host" "$commit"
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_all_parent_peak_evidence.py \
    --program "$program" --out "$output"
test -s "$output"

mapfile -t validation < <("$python" - "$output" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    result = json.load(stream)
if result.get("schema") != "ouruniv-cf4-all-parent-peak-evidence-result-v1":
    raise SystemExit("unexpected result schema")
status = result.get("status")
allowed = {
    "complete_pass_all_parent_peak_evidence_feasibility": True,
    "complete_fail_all_parent_peak_evidence_feasibility": False,
}
if status not in allowed:
    raise SystemExit("unexpected science status")
gates = result.get("gates", {})
decision = result.get("decision", {})
passed = bool(gates.get("feasibility_pass"))
if passed is not allowed[status]:
    raise SystemExit("status and feasibility gate disagree")
conditional = bool(decision.get("conditional_field_bank_authorized"))
candidate = bool(decision.get("candidate_generation_authorized"))
pm = bool(decision.get("PM_or_RAMSES_authorized"))
if conditional is not passed or candidate or pm:
    raise SystemExit("authorization flags violate the frozen firewall")
for value in (status, passed, conditional, candidate, pm):
    print(str(value).lower() if isinstance(value, bool) else value)
PY
)
if (( ${#validation[@]} != 5 )); then
    echo "result validation returned an incomplete record" >&2
    exit 65
fi
readonly science_status=${validation[0]}
readonly feasibility_pass=${validation[1]}
readonly conditional_bank_authorized=${validation[2]}
readonly candidate_authorized=${validation[3]}
readonly pm_authorized=${validation[4]}
printf '[runner] science_status=%s feasibility_pass=%s conditional_bank_authorized=%s\n' \
    "$science_status" "$feasibility_pass" "$conditional_bank_authorized"
