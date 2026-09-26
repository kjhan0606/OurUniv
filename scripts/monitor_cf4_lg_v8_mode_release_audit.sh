#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_audit
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
    echo "audit state or run log is absent" >&2
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
    local progress temporary
    temporary="$state/.MONITOR_HEARTBEAT.$$"
    progress=$(awk '/^\[(L0|metrics)\]/{line=$0} END{print line}' "$run_log")
    {
        printf 'checked_at=%s\nmarker=%s\n' "$(date --iso-8601=seconds)" "$marker"
        printf 'last_progress=%s\n' "${progress:-not_reported}"
    } >"$temporary"
    mv "$temporary" "$heartbeat"
}

while [[ ! -f "$state/COMPLETE" && ! -f "$state/FAILED" ]]; do
    write_heartbeat RUNNING
    sleep "$interval"
done
if [[ -f "$state/COMPLETE" ]]; then terminal=COMPLETE; else terminal=FAILED; fi
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
summary = {
    "schema": "ouruniv-cf4-v8-mode-release-audit-monitor-v1",
    "observed_at": datetime.now().astimezone().isoformat(),
    "terminal_marker": terminal,
    "terminal_marker_text": (state / terminal).read_text(),
}
result_path = state / "result.json"
if terminal == "COMPLETE" and result_path.is_file():
    result = json.loads(result_path.read_text())
    summary["status"] = result["status"]
    summary["gates"] = {
        name: result[name]["pass"] for name in ("L0", "L1", "L2", "L3", "L4", "L5")
        if name in result
    }
    summary["decision"] = result["decision"]
else:
    summary["run_log_tail"] = (state / "run.log").read_text(
        errors="replace"
    ).splitlines()[-40:]
temporary = output.with_name(f".{output.name}.{os.getpid()}")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
printf '[monitor] terminal=%s ended=%s\n' "$terminal" "$(date --iso-8601=seconds)"
