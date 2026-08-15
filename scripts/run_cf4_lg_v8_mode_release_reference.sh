#!/usr/bin/env bash
set -Eeuo pipefail

# Run the frozen V8 reference-only calibration without a Codex-attached
# process supervisor.  Launch this script through the companion tmux launcher.

umask 077

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/p2_lg_v8_cf4_mode_release_reference.json"
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference
readonly output="$state/calibration.json"
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly lock="$state/.runner.lock"
readonly environment="$state/environment.txt"

mkdir -p "$state"
exec 9>"$lock"
if ! flock -n 9; then
    echo "another V8 mode-release reference process owns $lock" >&2
    exit 75
fi

if [[ -e "$output" || -e "$running" || -e "$complete" || -e "$failed" \
      || -e "$log" || -e "$environment" ]]; then
    echo "refusing to overwrite an existing reference output or lifecycle marker" >&2
    exit 73
fi
if [[ ! -x "$python" || ! -f "$program" ]]; then
    echo "missing frozen Python environment or reference program" >&2
    exit 66
fi

exec >"$log" 2>&1
started_at=$(date --iso-8601=seconds)
host=$(hostname)
commit=$(git -C "$repo" rev-parse HEAD)
implementation_sha=$(
    sha256sum "$repo/src/cf4_lg_mode_release_reference.py" | awk '{print $1}'
)
program_sha=$(sha256sum "$program" | awk '{print $1}')
readonly started_at host commit implementation_sha program_sha

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo/src"

finish() {
    local rc=$?
    local ended_at
    local marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ -s "$output" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\n'
            printf 'started_at=%s\n' "$started_at"
            printf 'ended_at=%s\n' "$ended_at"
            printf 'host=%s\n' "$host"
            printf 'git_commit=%s\n' "$commit"
            printf 'output=%s\n' "$output"
            printf 'output_sha256=%s\n' "$(sha256sum "$output" | awk '{print $1}')"
            printf 'environment_sha256=%s\n' "$(sha256sum "$environment" | awk '{print $1}')"
        } >"$marker_tmp"
        mv "$marker_tmp" "$complete"
    else
        marker_tmp="$state/.FAILED.$$"
        {
            printf 'status=failed\n'
            printf 'exit_code=%s\n' "$rc"
            printf 'started_at=%s\n' "$started_at"
            printf 'ended_at=%s\n' "$ended_at"
            printf 'host=%s\n' "$host"
            printf 'git_commit=%s\n' "$commit"
            printf 'log=%s\n' "$log"
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
    printf 'status=running\n'
    printf 'pid=%s\n' "$$"
    printf 'started_at=%s\n' "$started_at"
    printf 'host=%s\n' "$host"
    printf 'git_commit=%s\n' "$commit"
    printf 'implementation_sha256=%s\n' "$implementation_sha"
    printf 'program_sha256=%s\n' "$program_sha"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-0}"
    printf 'log=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s gpu=%s\n' \
    "$started_at" "$host" "$$" "$commit" "${CUDA_VISIBLE_DEVICES:-0}"
printf '[runner] implementation_sha256=%s program_sha256=%s\n' \
    "$implementation_sha" "$program_sha"

cd "$repo"

"$python" - <<'PY' >"$environment"
import importlib.metadata
import platform

import jax
import numpy
import scipy

backend = jax.default_backend()
if backend != "gpu":
    raise SystemExit(f"reference calibration requires JAX GPU backend, got {backend!r}")

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"

print(f"python={platform.python_version()}")
print(f"jax={jax.__version__}")
print(f"jax_backend={backend}")
print(f"jax_devices={','.join(str(item) for item in jax.devices())}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"pmwd={version('pmwd')}")
PY

"$python" src/cf4_lg_mode_release_reference.py \
    --program "$program" \
    --out "$output"

test -s "$output"
