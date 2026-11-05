#!/usr/bin/env bash
set -Eeuo pipefail

readonly state=${CF4_ORACLE_REGRESSION_STATUS_STATE:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1_run}
readonly data=${CF4_ORACLE_REGRESSION_STATUS_DATA:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1}
marker_count=0
[[ -e "$state/COMPLETE" ]] && ((marker_count += 1))
[[ -e "$state/FAILED" ]] && ((marker_count += 1))
[[ -e "$state/RUNNING" ]] && ((marker_count += 1))

if (( marker_count > 1 )); then
    echo "status=invalid_state_conflicting_markers"
    exit 65
elif (( marker_count == 1 )) && [[ -e "$state/COMPLETE" && ! -s "$state/COMPLETE" \
      || -e "$state/FAILED" && ! -s "$state/FAILED" \
      || -e "$state/RUNNING" && ! -s "$state/RUNNING" ]]; then
    echo "status=invalid_state_empty_marker"
    exit 65
elif [[ -s "$state/COMPLETE" ]] && [[ ! -d "$data" \
      || ! -s "$data/arrays.npz" || ! -s "$data/result.json" \
      || ! -s "$data/manifest.json" ]]; then
    echo "status=invalid_complete_artifacts"
    exit 65
elif [[ -s "$state/COMPLETE" ]]; then
    cat "$state/COMPLETE"
elif [[ -s "$state/FAILED" ]]; then
    cat "$state/FAILED"
    [[ -s "$state/run.log" ]] && tail -n 20 "$state/run.log"
elif [[ -s "$state/RUNNING" ]]; then
    cat "$state/RUNNING"
    [[ -s "$state/run.log" ]] && tail -n 12 "$state/run.log"
elif [[ -e "$state" ]]; then
    echo "status=invalid_state_no_marker"
    exit 65
elif [[ -e "$data" ]]; then
    echo "status=invalid_data_without_lifecycle_state"
    exit 65
else
    echo "status=not_started"
fi
