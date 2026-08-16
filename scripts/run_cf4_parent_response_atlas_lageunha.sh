#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly source_commit=383b2f17b72d5baf510e5329db7c8a4b60d2681f
readonly runner="$repo/scripts/run_cf4_parent_response_atlas_lageunha.sh"
readonly program="$repo/config/cf4_parent_response_atlas_program.json"
readonly expected_program_sha=3a01a02728fe185112e2e58f4a9678796ce813a7884d939449ec6da418a6215a
readonly oracle="$repo/src/cf4_aggregate_evidence_oracle.py"
readonly expected_oracle_sha=3da4bd598f381e8a6fccd1dc2ae179cdd01e14d80ad0ad6c25dd1b3a93631d7f
readonly implementation="$repo/src/cf4_parent_response_atlas.py"
readonly expected_implementation_sha=5546965fe64951a95bde7ca486fe57f4601ad329558c8c878f126ef148c9d565
readonly phase_cache="$repo/src/cf4_peak_evidence_phase_cache.py"
readonly expected_phase_cache_sha=6359497a141aa0814c0dd663353ae6623f01e3676c0c2fdfea5a1b272e9d7106
readonly projection="$repo/src/cf4_projection_contract.py"
readonly expected_projection_sha=14ff16637980cf2c7565189b3e07bd899afe163ba71c6b1d123e70eb71a6f63f
readonly data_root=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1
readonly shards="$data_root/shards"
readonly manifest="$data_root/manifest.json"
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1_run
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly environment="$state/environment.txt"
readonly lock="$state/.runner.lock"

finish() {
    local rc=$? ended_at marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ -s "$manifest" && -n "${science_status:-}" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\nscience_status=%s\n' "$science_status"
            printf 'atlas_construction_pass=true\nparent_count=256\n'
            printf 'oracle_regression_authorized=false\nproduction_SMC_authorized=false\n'
            printf 'conditional_field_bank_authorized=false\nPM_or_RAMSES_authorized=false\n'
            printf 'started_at=%s\nended_at=%s\nhost=%s\ngit_commit=%s\n' \
                "$started_at" "$ended_at" "$host" "$commit"
            printf 'source_commit=%s\nprogram_sha256=%s\nrunner_sha256=%s\n' \
                "$source_commit" "$program_sha" "$runner_sha"
            printf 'oracle_sha256=%s\nimplementation_sha256=%s\n' \
                "$oracle_sha" "$implementation_sha"
            printf 'phase_cache_sha256=%s\nprojection_sha256=%s\n' \
                "$phase_cache_sha" "$projection_sha"
            printf 'manifest=%s\nmanifest_sha256=%s\n' \
                "$manifest" "$(sha256sum "$manifest" | awk '{print $1}')"
            printf 'shards=%s\nenvironment_sha256=%s\n' "$shards" \
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

host=$(hostname); readonly host
host_short=${host%%.*}; readonly host_short
if [[ "${host_short,,}" != "$expected_host" ]]; then
    echo "host gate failed: expected $expected_host, found $host" >&2; exit 69
fi
if [[ -e "$state" || -e "$data_root" ]]; then
    echo "refusing to reuse response-atlas state or data" >&2; exit 73
fi
if [[ ! -x "$python" || ! -f "$program" || ! -f "$oracle" \
      || ! -f "$implementation" || ! -f "$phase_cache" \
      || ! -f "$projection" ]]; then
    echo "missing response-atlas environment, program, or source" >&2; exit 66
fi

program_sha=$(sha256sum "$program" | awk '{print $1}')
oracle_sha=$(sha256sum "$oracle" | awk '{print $1}')
implementation_sha=$(sha256sum "$implementation" | awk '{print $1}')
phase_cache_sha=$(sha256sum "$phase_cache" | awk '{print $1}')
projection_sha=$(sha256sum "$projection" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
readonly program_sha oracle_sha implementation_sha phase_cache_sha projection_sha runner_sha
if [[ "$program_sha" != "$expected_program_sha" \
      || "$oracle_sha" != "$expected_oracle_sha" \
      || "$implementation_sha" != "$expected_implementation_sha" \
      || "$phase_cache_sha" != "$expected_phase_cache_sha" \
      || "$projection_sha" != "$expected_projection_sha" ]]; then
    echo "response-atlas program or source hash mismatch" >&2; exit 65
fi
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" HEAD; then
    echo "response-atlas source commit is not an ancestor of HEAD" >&2; exit 65
fi

readonly -a science_paths=(
    config/cf4_parent_response_atlas_program.json
    config/cf4_aggregate_evidence_annealed_smc_design.json
    src/cf4_aggregate_evidence_oracle.py
    src/cf4_parent_response_atlas.py
    src/cf4_peak_evidence_phase_cache.py
    src/cf4_projection_contract.py
    scripts/run_cf4_parent_response_atlas_lageunha.sh
)
for path in "${science_paths[@]}"; do
    git -C "$repo" ls-files --error-unmatch "$path" >/dev/null
done
if ! git -C "$repo" diff --quiet HEAD -- "${science_paths[@]}"; then
    echo "response-atlas science code/config differs from tracked HEAD" >&2; exit 65
fi

started_at=$(date --iso-8601=seconds)
commit=$(git -C "$repo" rev-parse HEAD)
readonly started_at commit
if [[ -e "$state" || -e "$data_root" ]]; then
    echo "response-atlas state or data appeared during preflight" >&2; exit 75
fi
mkdir "$state"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec 9>"$lock"
if ! flock -n 9; then
    echo "another response-atlas runner owns $lock" >&2; exit 75
fi

set -o noclobber
: >"$log"; : >"$environment"
set +o noclobber
exec >>"$log" 2>&1

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

running_tmp="$state/.RUNNING.$$"
{
    printf 'status=running\nstage=exact_parent_response_atlas\npid=%s\n' "$$"
    printf 'started_at=%s\nhost=%s\ngit_commit=%s\n' "$started_at" "$host" "$commit"
    printf 'source_commit=%s\nprogram_sha256=%s\nrunner_sha256=%s\n' \
        "$source_commit" "$program_sha" "$runner_sha"
    printf 'worker_processes=8\nthreads_per_worker=1\nlog=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s stage=response_atlas workers=8 threads=1\n' \
    "$started_at" "$host" "$$" "$commit"
{
    "$python" -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\n' "$host" "$commit"
} >>"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_parent_response_atlas.py --program "$program"
test -s "$manifest"

science_status=$(
    "$python" - "$manifest" \
        /gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference/calibration.json \
        "$shards" <<'PY'
import hashlib
import json
import pathlib
import sys

import numpy as np


EXPECTED_CALIBRATION_SHA256 = (
    "c9edb6d0a108746fe18fa75295ab73f53286a25f9aa2725d132a0560375cb988"
)
EXPECTED_CALIBRATION_STATUS = "complete_reference_calibration_parent3429_pass"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

manifest_path = pathlib.Path(sys.argv[1])
calibration_path = pathlib.Path(sys.argv[2])
shards_path = pathlib.Path(sys.argv[3]).resolve()
if sha256(calibration_path) != EXPECTED_CALIBRATION_SHA256:
    raise SystemExit("reference calibration hash mismatch during completion check")
calibration = json.loads(calibration_path.read_text())
if calibration.get("status") != EXPECTED_CALIBRATION_STATUS:
    raise SystemExit("reference calibration status mismatch during completion check")
expected_parents = calibration.get("reference_field_hashes", [])
if len(expected_parents) != 256:
    raise SystemExit("reference calibration parent count mismatch")
record = json.loads(manifest_path.read_text())
if record.get("schema") != "ouruniv-cf4-parent-response-atlas-manifest-v1" \
        or record.get("status") != "complete_exact_parent_response_atlas":
    raise SystemExit("unexpected response-atlas manifest contract")
if record.get("parent_count") != 256 or record.get("dtype") != "float64":
    raise SystemExit("response-atlas manifest parent count or dtype mismatch")
entries = record.get("entries", [])
if [row.get("seed") for row in entries] != list(range(3193, 3449)):
    raise SystemExit("response-atlas manifest seed lineage mismatch")
atlas_paths = []
for row, parent in zip(entries, expected_parents, strict=True):
    if (
        row.get("seed") != parent.get("seed")
        or row.get("parent_field") != parent.get("path")
        or row.get("parent_field_sha256") != parent.get("sha256")
    ):
        raise SystemExit("response-atlas parent lineage differs from calibration")
    path = pathlib.Path(row.get("atlas", ""))
    if row.get("shape") != [101, 101, 101] or row.get("dtype") != "float64" \
            or not path.is_file() or path.resolve().parent != shards_path \
            or sha256(path) != row.get("atlas_sha256"):
        raise SystemExit("response-atlas shard lineage mismatch")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (101, 101, 101) or value.dtype != np.dtype("float64") \
            or not np.all(np.isfinite(value)):
        raise SystemExit("response-atlas shard header or finite gate failed")
    atlas_paths.append(path.resolve())
if len(set(atlas_paths)) != 256:
    raise SystemExit("response-atlas shard paths are not unique")
firewall = record.get("information_firewall", {})
if any(firewall.get(key) is not False for key in (
        "CF4_deviance_loaded", "production_SMC_particles_loaded",
        "old_adaptation_cache_imported")):
    raise SystemExit("response-atlas information firewall opened")
decision = record.get("decision", {})
if decision.get("atlas_construction_pass") is not True:
    raise SystemExit("response-atlas construction did not pass")
for key in (
        "production_SMC_authorized", "conditional_field_bank_authorized",
        "PM_or_RAMSES_authorized"):
    if decision.get(key) is not False:
        raise SystemExit("response-atlas manifest opened a forbidden action")
print(record["status"])
PY
)
readonly science_status
printf '[runner] science_status=%s manifest=%s\n' "$science_status" "$manifest"
