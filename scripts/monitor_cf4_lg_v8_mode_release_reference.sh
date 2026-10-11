#!/usr/bin/env bash
set -Eeuo pipefail

# Low-frequency, marker-only monitor for the detached V8 reference job.
# It never inspects process tables and exits after writing one terminal summary.

umask 077
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference
readonly run_log="$state/run.log"
readonly heartbeat="$state/MONITOR_HEARTBEAT"
readonly summary="$state/monitor_summary.json"
readonly monitor_log="$state/monitor.log"
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly interval=${MONITOR_INTERVAL_SECONDS:-300}

if [[ ! "$interval" =~ ^[1-9][0-9]*$ ]]; then
    echo "MONITOR_INTERVAL_SECONDS must be a positive integer" >&2
    exit 64
fi
if [[ ! -d "$state" || ! -f "$run_log" ]]; then
    echo "reference state or run log is absent" >&2
    exit 66
fi
if [[ -e "$summary" || -e "$monitor_log" ]]; then
    echo "monitor output already exists; refusing to overwrite it" >&2
    exit 73
fi

exec >"$monitor_log" 2>&1
printf '[monitor] started=%s interval_seconds=%s\n' \
    "$(date --iso-8601=seconds)" "$interval"

write_heartbeat() {
    local marker=$1
    local progress
    local temporary="$state/.MONITOR_HEARTBEAT.$$"
    progress=$(awk '/^\[reference\] [0-9]+\/256$/{line=$0} END{print line}' "$run_log")
    {
        printf 'checked_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'marker=%s\n' "$marker"
        printf 'last_progress=%s\n' "${progress:-not_reported}"
    } >"$temporary"
    mv "$temporary" "$heartbeat"
}

while [[ ! -f "$state/COMPLETE" && ! -f "$state/FAILED" ]]; do
    write_heartbeat RUNNING
    sleep "$interval"
done

if [[ -f "$state/COMPLETE" ]]; then
    terminal=COMPLETE
else
    terminal=FAILED
fi
write_heartbeat "$terminal"

"$python" - "$state" "$terminal" "$summary" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

state = Path(sys.argv[1])
terminal = sys.argv[2]
output = Path(sys.argv[3])
report = {
    "schema": "ouruniv-cf4-v8-reference-monitor-summary-v1",
    "observed_at": datetime.now().astimezone().isoformat(),
    "terminal_marker": terminal,
    "terminal_marker_text": (state / terminal).read_text(),
}
calibration = state / "calibration.json"
if terminal == "COMPLETE" and calibration.is_file():
    result = json.loads(calibration.read_text())
    report.update({
        "calibration_status": result["status"],
        "L2_parent3429": result["L2_parent3429"],
        "two_chain_all_pass": result["two_chain_audit"]["all_pass"],
        "decision": result["decision"],
    })
else:
    lines = (state / "run.log").read_text(errors="replace").splitlines()
    report["run_log_tail"] = lines[-40:]

temporary = output.with_name(f".{output.name}.{os.getpid()}")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY

printf '[monitor] terminal=%s summary=%s ended=%s\n' \
    "$terminal" "$summary" "$(date --iso-8601=seconds)"
