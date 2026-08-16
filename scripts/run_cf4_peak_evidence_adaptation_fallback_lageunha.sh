#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly runner="$repo/scripts/run_cf4_peak_evidence_adaptation_fallback_lageunha.sh"
readonly program="$repo/config/cf4_peak_evidence_adaptation_fallback_program.json"
readonly expected_program_sha=1d3451103ea9e9c01fd60a8276a0ada713fd652c1ee2aea4f196036443e581b0
readonly implementation="$repo/src/cf4_peak_evidence_adaptation_fallback.py"
readonly expected_implementation_sha=b8a3a66d3ef4916cc157a70379ec7eaee8d336178e83870ab342681d1f2c3c4a
readonly core="$repo/src/cf4_peak_evidence_adaptation.py"
readonly expected_core_sha=b5aa7b4f5c976532647ad0583e62e3395575cabed03a9e10199e5d7704f9424d
readonly proposal_implementation="$repo/src/cf4_adaptive_geometry_proposal.py"
readonly expected_proposal_implementation_sha=403eadcd601bcff14c8cf7f2f438b6dad4640d8be08752178e5cd7b709ec93cc
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_adaptation_fallback_v1
readonly output="$state/result.json"
readonly arrays="$state/arrays.npz"
readonly proposal="$state/proposal.json"
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly environment="$state/environment.txt"
readonly lock="$state/.runner.lock"

mkdir -p "$state"
exec 9>"$lock"
if ! flock -n 9; then echo "another fallback runner owns $lock" >&2; exit 75; fi

host=$(hostname); readonly host
host_short=${host%%.*}; readonly host_short
if [[ "${host_short,,}" != "$expected_host" ]]; then
    echo "host gate failed: expected $expected_host, found $host" >&2; exit 69
fi
if [[ -e "$output" || -e "$arrays" || -e "$proposal" || -e "$running" \
      || -e "$complete" || -e "$failed" || -e "$log" || -e "$environment" ]]; then
    echo "refusing to overwrite a fallback adaptation file" >&2; exit 73
fi
if [[ ! -x "$python" || ! -f "$program" || ! -f "$implementation" \
      || ! -f "$core" || ! -f "$proposal_implementation" ]]; then
    echo "missing fallback environment, program, or implementation" >&2; exit 66
fi

program_sha=$(sha256sum "$program" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
core_sha=$(sha256sum "$core" | awk '{print $1}')
proposal_implementation_sha=$(sha256sum "$proposal_implementation" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
readonly program_sha implementation_sha core_sha proposal_implementation_sha runner_sha
if [[ "$program_sha" != "$expected_program_sha" \
      || "$implementation_sha" != "$expected_implementation_sha" \
      || "$core_sha" != "$expected_core_sha" \
      || "$proposal_implementation_sha" != "$expected_proposal_implementation_sha" ]]; then
    echo "fallback program or implementation hash mismatch" >&2; exit 65
fi

readonly -a science_paths=(
    config/cf4_peak_evidence_adaptation_fallback_program.json
    config/cf4_peak_evidence_adaptation_v1_result_record.json
    config/cf4_peak_evidence_adaptive_integration_design.json
    config/p2_lg_z0_forward_importance_v8.json
    src/cf4_peak_evidence_adaptation_fallback.py
    src/cf4_peak_evidence_adaptation.py
    src/cf4_adaptive_geometry_proposal.py
    src/cf4_all_parent_peak_evidence.py
    src/cf4_peak_evidence_phase_cache.py
    src/cf4_lg_peak_cr.py
    scripts/run_cf4_peak_evidence_adaptation_fallback_lageunha.sh
)
for path in "${science_paths[@]}"; do
    git -C "$repo" ls-files --error-unmatch "$path" >/dev/null
done
if ! git -C "$repo" diff --quiet HEAD -- "${science_paths[@]}"; then
    echo "fallback science code/config differs from tracked HEAD" >&2; exit 65
fi

set -o noclobber
: >"$log"; : >"$environment"
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
    if (( rc == 0 )) && [[ -s "$output" && -s "$arrays" \
                          && -n "${science_status:-}" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\nscience_status=%s\nadaptation_stage=fallback_8192\nadaptation_pass=%s\n' \
                "$science_status" "$adaptation_pass"
            printf 'final_proposal_frozen=%s\nindependent_8192_final_bank_authorized=%s\n' \
                "$proposal_frozen" "$final_authorized"
            printf 'additional_adaptation_fallback_authorized=false\n'
            printf 'conditional_field_bank_authorized=false\ncandidate_generation_authorized=false\nPM_or_RAMSES_authorized=false\n'
            printf 'started_at=%s\nended_at=%s\nhost=%s\ngit_commit=%s\n' \
                "$started_at" "$ended_at" "$host" "$commit"
            printf 'program_sha256=%s\nimplementation_sha256=%s\nadaptation_core_sha256=%s\n' \
                "$program_sha" "$implementation_sha" "$core_sha"
            printf 'proposal_implementation_sha256=%s\nrunner_sha256=%s\n' \
                "$proposal_implementation_sha" "$runner_sha"
            printf 'output=%s\noutput_sha256=%s\narrays=%s\narrays_sha256=%s\n' \
                "$output" "$(sha256sum "$output" | awk '{print $1}')" \
                "$arrays" "$(sha256sum "$arrays" | awk '{print $1}')"
            if [[ "$proposal_frozen" == true ]]; then
                printf 'proposal=%s\nproposal_sha256=%s\n' "$proposal" \
                    "$(sha256sum "$proposal" | awk '{print $1}')"
            fi
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
    printf 'status=running\nadaptation_stage=fallback_8192\npid=%s\nstarted_at=%s\n' "$$" "$started_at"
    printf 'host=%s\ngit_commit=%s\nprogram_sha256=%s\n' "$host" "$commit" "$program_sha"
    printf 'implementation_sha256=%s\nadaptation_core_sha256=%s\n' "$implementation_sha" "$core_sha"
    printf 'proposal_implementation_sha256=%s\nrunner_sha256=%s\n' \
        "$proposal_implementation_sha" "$runner_sha"
    printf 'worker_processes=8\nthreads_per_worker=1\nlog=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s stage=fallback_8192 workers=8 threads=1\n' \
    "$started_at" "$host" "$$" "$commit"
printf '[runner] runner_sha256=%s\n' "$runner_sha"
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\n' "$host" "$commit"
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_peak_evidence_adaptation_fallback.py \
    --program "$program" --out "$output" --arrays-out "$arrays" \
    --proposal-out "$proposal"
test -s "$output"; test -s "$arrays"

mapfile -t validation < <("$python" - "$output" "$arrays" "$proposal" <<'PY'
import hashlib
import json
import pathlib
import sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

output, arrays, proposal = map(pathlib.Path, sys.argv[1:])
with output.open() as stream:
    result = json.load(stream)
if result.get("schema") != "ouruniv-cf4-peak-evidence-adaptation-fallback-result-v1" \
        or result.get("adaptation_stage") != "fallback_8192":
    raise SystemExit("unexpected fallback adaptation result contract")
status = result.get("status")
allowed = {
    "complete_pass_freeze_defensive_final_proposal_from_fallback": True,
    "complete_fail_fallback_adaptation": False,
}
if status not in allowed:
    raise SystemExit("unexpected fallback science status")
passed = bool(result.get("gates", {}).get("adaptation_pass"))
decision = result.get("decision", {})
proposal_frozen = bool(decision.get("final_proposal_frozen"))
final_authorized = bool(decision.get("independent_8192_final_bank_authorized"))
if passed is not allowed[status] or proposal_frozen is not passed \
        or final_authorized is not passed:
    raise SystemExit("fallback status and final-proposal authorization disagree")
if decision.get("fallback_8192_adaptation_bank_authorized") is not False \
        or decision.get("additional_adaptation_fallback_authorized") is not False:
    raise SystemExit("fallback result opened a recursive adaptation")
for key in ("conditional_field_bank_authorized", "candidate_generation_authorized",
            "parent_or_seed_selection_authorized", "PM_or_RAMSES_authorized"):
    if decision.get(key) is not False:
        raise SystemExit("fallback opened a forbidden downstream action")
lineage = result.get("lineage", {})
if pathlib.Path(lineage.get("arrays", "")) != arrays \
        or lineage.get("arrays_sha256") != sha256(arrays):
    raise SystemExit("fallback array lineage mismatch")
if passed:
    if not proposal.is_file() or pathlib.Path(lineage.get("proposal", "")) != proposal \
            or lineage.get("proposal_sha256") != sha256(proposal):
        raise SystemExit("fallback proposal lineage mismatch")
elif proposal.exists():
    raise SystemExit("failed fallback unexpectedly wrote a proposal")
for value in (status, passed, proposal_frozen, final_authorized):
    print(str(value).lower() if isinstance(value, bool) else value)
PY
)
if (( ${#validation[@]} != 4 )); then
    echo "fallback result validation returned an incomplete record" >&2; exit 65
fi
readonly science_status=${validation[0]}
readonly adaptation_pass=${validation[1]}
readonly proposal_frozen=${validation[2]}
readonly final_authorized=${validation[3]}
printf '[runner] science_status=%s adaptation_pass=%s final_authorized=%s\n' \
    "$science_status" "$adaptation_pass" "$final_authorized"
