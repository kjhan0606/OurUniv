#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly source_commit=05b2cc373c94ba719492929f10409383257100b1
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_oracle_regression_lageunha.sh"
readonly program="$repo/config/cf4_aggregate_evidence_oracle_regression_program.json"
readonly expected_program_sha=0576851112f83d9566006d1b9afca37eceebb8b21989ed87665fa8e6ffa30016
readonly design="$repo/config/cf4_aggregate_evidence_oracle_regression_design.json"
readonly expected_design_sha=b735918021d6898bfbb4b29f7f4f3f732fea2804205a813761a52cc4b7616dd0
readonly implementation="$repo/src/cf4_aggregate_evidence_oracle_regression.py"
readonly expected_implementation_sha=5e26b43a9560f9eab2ecd706e2062e99a307a77ba16aba654cbd2967fe5b05f0
readonly tests="$repo/tests/test_cf4_aggregate_evidence_oracle_regression.py"
readonly expected_tests_sha=289fdfe7fa7da1a8a47b147d0cf436dc10eb6d280acea2f7817f4d9e3baf2eb3
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1
readonly arrays="$data/arrays.npz"
readonly result="$data/result.json"
readonly manifest="$data/manifest.json"
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1_run
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly environment="$state/environment.txt"
readonly lock="$state/.runner.lock"

finish() {
    local rc=$? ended_at marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ -s "$manifest" && -s "$result" && -s "$arrays" \
          && -n "${science_status:-}" && -n "${oracle_pass:-}" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\nscience_status=%s\n' "$science_status"
            printf 'oracle_regression_pass=%s\nfailure_class=%s\n' \
                "$oracle_pass" "$failure_class"
            printf 'production_SMC_program_design_authorized=%s\n' \
                "$production_design"
            printf 'production_SMC_execution_authorized=false\n'
            printf 'conditional_field_bank_authorized=false\n'
            printf 'parent_or_seed_selection_authorized=false\n'
            printf 'PM_or_halo_finder_authorized=false\nRAMSES_authorized=false\n'
            printf 'started_at=%s\nended_at=%s\nhost=%s\ngit_commit=%s\n' \
                "$started_at" "$ended_at" "$host" "$commit"
            printf 'source_commit=%s\nprogram_sha256=%s\nrunner_sha256=%s\n' \
                "$source_commit" "$program_sha" "$runner_sha"
            printf 'design_sha256=%s\nimplementation_sha256=%s\n' \
                "$design_sha" "$implementation_sha"
            printf 'tests_sha256=%s\nmanifest=%s\nmanifest_sha256=%s\n' \
                "$tests_sha" "$manifest" "$(sha256sum "$manifest" | awk '{print $1}')"
            printf 'result_sha256=%s\narrays_sha256=%s\n' \
                "$(sha256sum "$result" | awk '{print $1}')" \
                "$(sha256sum "$arrays" | awk '{print $1}')"
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

validate_program_and_pins() {
    env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from cf4_aggregate_evidence_oracle_regression import validate_program

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

program_path = Path(sys.argv[1]).resolve()
program = json.loads(program_path.read_text())
validate_program(program, program_path)
root = program_path.parents[1]
for item in program["pinned_local_files"]:
    path = root / item["path"]
    if sha256(path) != item["sha256"]:
        raise SystemExit(f"pinned local hash mismatch: {item['path']}")
PY
}

host=$(hostname); readonly host
host_short=${host%%.*}; readonly host_short
if [[ "${host_short,,}" != "$expected_host" ]]; then
    echo "host gate failed: expected $expected_host, found $host" >&2; exit 69
fi
if [[ -e "$state" || -e "$data" ]]; then
    echo "refusing to reuse oracle-regression state or data" >&2; exit 73
fi
if [[ ! -x "$python" || ! -f "$program" || ! -f "$design" \
      || ! -f "$implementation" || ! -f "$tests" ]]; then
    echo "missing oracle-regression environment, program, source, or tests" >&2; exit 66
fi

program_sha=$(sha256sum "$program" | awk '{print $1}')
design_sha=$(sha256sum "$design" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
tests_sha=$(sha256sum "$tests" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
readonly program_sha design_sha implementation_sha tests_sha runner_sha
if [[ "$program_sha" != "$expected_program_sha" \
      || "$design_sha" != "$expected_design_sha" \
      || "$implementation_sha" != "$expected_implementation_sha" \
      || "$tests_sha" != "$expected_tests_sha" ]]; then
    echo "oracle-regression program, design, source, or test hash mismatch" >&2; exit 65
fi
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" HEAD; then
    echo "oracle-regression source commit is not an ancestor of HEAD" >&2; exit 65
fi

readonly -a science_paths=(
    config/cf4_aggregate_evidence_oracle_regression_program.json
    config/cf4_aggregate_evidence_oracle_regression_design.json
    src/cf4_aggregate_evidence_oracle_regression.py
    tests/test_cf4_aggregate_evidence_oracle_regression.py
    src/cf4_aggregate_evidence_oracle.py
    src/cf4_peak_evidence_phase_cache.py
    src/cf4_projection_contract.py
    src/cf4_peak_evidence.py
    src/cf4_lg_peak_cr.py
    config/cf4_parent_response_atlas_v1_result_record.json
    config/cf4_peak_evidence_phase_control_v2_result_record.json
    config/cf4_peak_evidence_adaptation_v1_result_record.json
    config/cf4_peak_evidence_adaptation_fallback_result_record.json
    config/p2_lg_z0_forward_importance_v8.json
    scripts/run_cf4_aggregate_evidence_oracle_regression_lageunha.sh
)
for path in "${science_paths[@]}"; do
    git -C "$repo" ls-files --error-unmatch "$path" >/dev/null
done
if ! git -C "$repo" diff --quiet HEAD -- "${science_paths[@]}"; then
    echo "oracle-regression science code/config differs from tracked HEAD" >&2; exit 65
fi

validate_program_and_pins

started_at=$(date --iso-8601=seconds)
commit=$(git -C "$repo" rev-parse HEAD)
readonly started_at commit
if [[ -e "$state" || -e "$data" ]]; then
    echo "oracle-regression state or data appeared during preflight" >&2; exit 75
fi
mkdir "$state"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec 9>"$lock"
if ! flock -n 9; then
    echo "another oracle-regression runner owns $lock" >&2; exit 75
fi

set -o noclobber
: >"$log"; : >"$environment"
set +o noclobber
exec >>"$log" 2>&1

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

running_tmp="$state/.RUNNING.$$"
{
    printf 'status=running\nstage=exact_aggregate_evidence_oracle_regression\npid=%s\n' "$$"
    printf 'started_at=%s\nhost=%s\ngit_commit=%s\n' "$started_at" "$host" "$commit"
    printf 'source_commit=%s\nprogram_sha256=%s\nrunner_sha256=%s\n' \
        "$source_commit" "$program_sha" "$runner_sha"
    printf 'worker_processes=8\nthreads_per_worker=1\nlog=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s stage=oracle_regression workers=8 threads=1\n' \
    "$started_at" "$host" "$$" "$commit"
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\n' "$host" "$commit"
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_aggregate_evidence_oracle_regression.py \
    --program "$program"
test -s "$manifest"; test -s "$result"; test -s "$arrays"

if [[ "$(sha256sum "$program" | awk '{print $1}')" != "$program_sha" \
      || "$(sha256sum "$design" | awk '{print $1}')" != "$design_sha" \
      || "$(sha256sum "$implementation" | awk '{print $1}')" != "$implementation_sha" \
      || "$(sha256sum "$tests" | awk '{print $1}')" != "$tests_sha" \
      || "$(sha256sum "$runner" | awk '{print $1}')" != "$runner_sha" ]]; then
    echo "oracle-regression postflight source hash mismatch" >&2; exit 65
fi
if [[ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]] \
      || ! git -C "$repo" diff --quiet HEAD -- "${science_paths[@]}"; then
    echo "oracle-regression HEAD or science paths changed during execution" >&2; exit 65
fi
validate_program_and_pins

postcheck=$(
    "$python" - "$manifest" "$result" "$arrays" "$program" "$design" \
        "$program_sha" "$design_sha" "$implementation_sha" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def array_sha(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True

manifest_path, result_path, arrays_path, program_path, design_path = map(
    Path, sys.argv[1:6]
)
expected_program_sha, expected_design_sha, expected_implementation_sha = sys.argv[6:9]
manifest = json.loads(manifest_path.read_text())
result = json.loads(result_path.read_text())
program = json.loads(program_path.read_text())
design = json.loads(design_path.read_text())
allowed_status = {
    "complete_pass_exact_oracle_regression",
    "complete_fail_exact_oracle_regression",
}
if manifest.get("schema") != "ouruniv-cf4-aggregate-evidence-oracle-regression-manifest-v1" \
        or result.get("schema") != design["result_schema"]["schema"] \
        or manifest.get("status") != result.get("status") \
        or result.get("status") not in allowed_status:
    raise SystemExit("oracle-regression result or manifest contract mismatch")
if manifest.get("manifest") != str(manifest_path) \
        or manifest.get("program") != str(program_path) \
        or manifest.get("arrays") != str(arrays_path) \
        or manifest.get("result") != str(result_path) \
        or sha256(program_path) != expected_program_sha \
        or sha256(design_path) != expected_design_sha \
        or program["design"]["sha256"] != expected_design_sha \
        or program["implementation"]["sha256"] != expected_implementation_sha \
        or manifest.get("program_sha256") != expected_program_sha \
        or manifest.get("implementation_sha256") != expected_implementation_sha \
        or manifest.get("arrays_sha256") != sha256(arrays_path) \
        or manifest.get("result_sha256") != sha256(result_path):
    raise SystemExit("oracle-regression artifact lineage mismatch")
lineage = result.get("lineage", {})
if lineage.get("program") != str(program_path) \
        or lineage.get("program_sha256") != expected_program_sha \
        or lineage.get("design") != str(design_path) \
        or lineage.get("design_sha256") != expected_design_sha \
        or lineage.get("implementation_sha256") != expected_implementation_sha \
        or lineage.get("arrays") != str(arrays_path) \
        or lineage.get("arrays_sha256") != sha256(arrays_path):
    raise SystemExit("oracle-regression result lineage mismatch")
if sorted(path.name for path in manifest_path.parent.iterdir()) != [
        "arrays.npz", "manifest.json", "result.json"]:
    raise SystemExit("oracle-regression data directory has unexpected files")

actual_contract = {}
with np.load(arrays_path, allow_pickle=False) as item:
    if set(item.files) != set(design["arrays_contract"]):
        raise SystemExit("oracle-regression arrays key set mismatch")
    for key, (dtype, shape) in design["arrays_contract"].items():
        value = item[key]
        if str(value.dtype) != dtype or list(value.shape) != shape \
                or not np.all(np.isfinite(value)):
            raise SystemExit(f"oracle-regression array contract mismatch: {key}")
        actual_contract[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    inside_selection_sha = array_sha(
        item["inside_keys"], item["inside_candidate_index"]
    )
    outside_selection_sha = array_sha(
        item["outside_keys"], item["outside_candidate_index"]
    )
if manifest.get("arrays_shape_dtype") != actual_contract:
    raise SystemExit("oracle-regression manifest array metadata mismatch")
if inside_selection_sha != "d91f20f27fb2f43df78179b2e9f89bd9c29f802a41a107ca69ab5d375955b13" \
        or outside_selection_sha != "3b5ab76f91b008a8671b33a46f33f36f4788251bebe53e35ce784cabc031aefc" \
        or result["selection"]["inside_selection_sha256"] != inside_selection_sha \
        or result["selection"]["outside_selection_sha256"] != outside_selection_sha:
    raise SystemExit("oracle-regression deterministic selection mismatch")
if not set(design["result_schema"]["required_sections"]).issubset(result) \
        or not all_finite(result):
    raise SystemExit("oracle-regression result sections or finite gate failed")
gates = result.get("gates", {})
decision = result.get("decision", {})
passed = gates.get("oracle_regression_pass") is True
if gates.get("all_lineage_and_input_contracts") is not True \
        or gates.get("all_values_finite") is not True \
        or decision.get("oracle_regression_pass") is not passed:
    raise SystemExit("oracle-regression validity or decision gate failed")
if passed:
    if result["status"] != "complete_pass_exact_oracle_regression" \
            or result.get("failure_class") is not None \
            or decision.get("production_SMC_program_design_authorized") is not True:
        raise SystemExit("oracle-regression pass classification mismatch")
else:
    allowed_failure = {
        "dense_27_phase_mismatch",
        "inside_atlas_mismatch",
        "outside_slow_path_or_lineage_mismatch",
        "historical_2048_mismatch",
        "historical_8192_mismatch",
    }
    if result["status"] != "complete_fail_exact_oracle_regression" \
            or result.get("failure_class") not in allowed_failure \
            or decision.get("production_SMC_program_design_authorized") is not False:
        raise SystemExit("oracle-regression scientific-fail classification mismatch")
for key in (
        "production_SMC_execution_authorized",
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
        "automatic_follow_on"):
    if decision.get(key) is not False:
        raise SystemExit(f"oracle-regression opened forbidden decision: {key}")
if result.get("information_firewall") != design["information_firewall"] \
        or manifest.get("information_firewall") != design["information_firewall"]:
    raise SystemExit("oracle-regression information firewall mismatch")
if manifest.get("gates") != gates or manifest.get("decision") != decision:
    raise SystemExit("oracle-regression manifest summary mismatch")
print(result["status"] + "\t" + str(passed).lower() + "\t"
      + (result.get("failure_class") or "none") + "\t"
      + str(decision["production_SMC_program_design_authorized"]).lower())
PY
)
IFS=$'\t' read -r science_status oracle_pass failure_class production_design \
    <<<"$postcheck"
readonly science_status oracle_pass failure_class production_design
if [[ -z "$science_status" || -z "$oracle_pass" || -z "$failure_class" \
      || -z "$production_design" ]]; then
    echo "oracle-regression completion validator returned an incomplete summary" >&2; exit 65
fi
printf '[runner] science_status=%s oracle_regression_pass=%s failure_class=%s\n' \
    "$science_status" "$oracle_pass" "$failure_class"
