#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_peak_evidence_phase_control_v2_program.json"
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_phase_control_v2
readonly output="$state/result.json"
readonly filter="$state/density_filter_rfft.npy"
readonly log="$state/run.log"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"
readonly environment="$state/environment.txt"
readonly lock="$state/.runner.lock"

mkdir -p "$state"
exec 9>"$lock"
if ! flock -n 9; then echo "another v2 phase control owns $lock" >&2; exit 75; fi
if [[ -e "$output" || -e "$filter" || -e "$running" || -e "$complete" \
      || -e "$failed" || -e "$log" || -e "$environment" ]]; then
    echo "refusing to overwrite a v2 phase-control file" >&2
    exit 73
fi
if [[ ! -x "$python" || ! -f "$program" ]]; then
    echo "missing Python environment or v2 program" >&2
    exit 66
fi

exec >"$log" 2>&1
started_at=$(date --iso-8601=seconds)
host=$(hostname)
commit=$(git -C "$repo" rev-parse HEAD)
implementation_sha=$(sha256sum "$repo/src/cf4_peak_evidence_phase_control.py" | awk '{print $1}')
program_sha=$(sha256sum "$program" | awk '{print $1}')
readonly started_at host commit implementation_sha program_sha

export PYTHONNOUSERSITE=1 PYTHONPATH="$repo/src" JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

finish() {
    local rc=$? ended_at marker_tmp
    ended_at=$(date --iso-8601=seconds)
    if (( rc == 0 )) && [[ -s "$output" && -s "$filter" ]]; then
        marker_tmp="$state/.COMPLETE.$$"
        {
            printf 'status=complete\nstarted_at=%s\nended_at=%s\n' "$started_at" "$ended_at"
            printf 'host=%s\ngit_commit=%s\n' "$host" "$commit"
            printf 'output=%s\noutput_sha256=%s\n' "$output" "$(sha256sum "$output" | awk '{print $1}')"
            printf 'filter=%s\nfilter_sha256=%s\n' "$filter" "$(sha256sum "$filter" | awk '{print $1}')"
            printf 'environment_sha256=%s\n' "$(sha256sum "$environment" | awk '{print $1}')"
        } >"$marker_tmp"
        mv "$marker_tmp" "$complete"
    else
        marker_tmp="$state/.FAILED.$$"
        {
            printf 'status=failed\nexit_code=%s\nstarted_at=%s\nended_at=%s\n' "$rc" "$started_at" "$ended_at"
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
    printf 'implementation_sha256=%s\nprogram_sha256=%s\n' "$implementation_sha" "$program_sha"
    printf 'thread_limit=1\njax_platform=cpu\nlog=%s\n' "$log"
} >"$running_tmp"
mv "$running_tmp" "$running"

printf '[runner] start=%s host=%s pid=%s commit=%s threads=1 jax=cpu\n' "$started_at" "$host" "$$" "$commit"
printf '[runner] implementation_sha256=%s program_sha256=%s\n' "$implementation_sha" "$program_sha"
{
    "$python" -c 'import platform, jax, numpy, scipy; print(f"python={platform.python_version()}"); print(f"jax={jax.__version__}"); print(f"jax_backend={jax.default_backend()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
    printf 'host=%s\ncommit=%s\n' "$host" "$commit"
} >"$environment"

cd "$repo"
nice -n 5 "$python" src/cf4_peak_evidence_phase_control.py \
    --program "$program" --out "$output" --filter-out "$filter"
test -s "$output"; test -s "$filter"
