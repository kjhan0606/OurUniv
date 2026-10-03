#!/usr/bin/env bash
set -Eeuo pipefail

readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1_run
marker_count=0
[[ -s "$state/COMPLETE" ]] && ((marker_count += 1))
[[ -s "$state/FAILED" ]] && ((marker_count += 1))
[[ -s "$state/RUNNING" ]] && ((marker_count += 1))

if (( marker_count > 1 )); then
    echo "status=invalid_state_conflicting_markers"
    exit 65
elif [[ -s "$state/COMPLETE" ]]; then
    cat "$state/COMPLETE"
elif [[ -s "$state/FAILED" ]]; then
    cat "$state/FAILED"
    [[ -s "$state/run.log" ]] && tail -n 20 "$state/run.log"
elif [[ -s "$state/RUNNING" ]]; then
    cat "$state/RUNNING"
    [[ -s "$state/run.log" ]] && tail -n 8 "$state/run.log"
elif [[ -e "$state" ]]; then
    echo "status=invalid_state_no_marker"
    exit 65
else
    echo "status=not_started"
fi
