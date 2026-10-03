#!/usr/bin/env bash
# Run the EAGLE particle pass with a lock, durable log, and exit marker.

set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${EAGLE_ROOT:-/gpfs/kjhan/EAGLE/RefL0100N1504}
log_dir=$root/logs
status=$root/prepare_ref100_hong.status
run_id=$(date +%Y%m%dT%H%M%S%z)
log=$log_dir/prepare_ref100_hong_${run_id}.log

mkdir -p "$log_dir"
exec 9>"$root/.prepare_ref100_hong.lock"
if ! flock -n 9; then
    printf 'Another EAGLE preparation process holds %s\n' \
        "$root/.prepare_ref100_hong.lock" >&2
    exit 3
fi

printf 'running\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$(hostname)" "$log" >"$status"
printf '[start] host=%s time=%s log=%s\n' \
    "$(hostname)" "$(date --iso-8601=seconds)" "$log" | tee "$log"

set +e
PYTHONUNBUFFERED=1 python "$repo/src/hong2021_prepare_eagle.py" "$@" \
    2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -eq 0 ]]; then
    state=complete
else
    state=failed
fi
printf '%s\t%s\t%s\t%s\texit=%s\n' \
    "$state" "$(date --iso-8601=seconds)" "$(hostname)" "$log" "$rc" \
    >"$status"
printf '[%s] host=%s time=%s exit=%s\n' \
    "$state" "$(hostname)" "$(date --iso-8601=seconds)" "$rc" | tee -a "$log"
exit "$rc"
