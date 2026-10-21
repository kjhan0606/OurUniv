#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly state=${CF4_AGGREGATE_SMC_STATUS_STATE:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1_run}
readonly data=${CF4_AGGREGATE_SMC_STATUS_DATA:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1}

if [[ ! -e "$state" && ! -e "$data" ]]; then
    echo "status=not_started_fail_closed"
    exit 3
fi
if [[ -e "$state" && ! -d "$state" ]]; then
    echo "status=invalid_state_not_directory"
    exit 65
fi
if [[ -e "$data" && ! -d "$data" ]]; then
    echo "status=invalid_data_not_directory"
    exit 65
fi
if [[ -d "$state" && ! -d "$data" ]]; then
    echo "status=invalid_orphan_state_without_data"
    exit 65
fi
if [[ -d "$data" && ! -d "$state" ]]; then
    echo "status=invalid_orphan_data_without_state"
    exit 65
fi

marker_count=0
[[ -e "$state/RUNNING" ]] && ((marker_count += 1))
[[ -e "$state/COMPLETE" ]] && ((marker_count += 1))
[[ -e "$state/FAILED" ]] && ((marker_count += 1))
if (( marker_count > 1 )); then
    echo "status=invalid_state_conflicting_markers"
    exit 65
fi
if (( marker_count == 0 )); then
    echo "status=invalid_state_markerless"
    exit 65
fi
if [[ -e "$state/RUNNING" && ! -s "$state/RUNNING" \
      || -e "$state/COMPLETE" && ! -s "$state/COMPLETE" \
      || -e "$state/FAILED" && ! -s "$state/FAILED" ]]; then
    echo "status=invalid_state_empty_marker"
    exit 65
fi

if [[ -s "$state/COMPLETE" ]]; then
    if [[ ! -s "$data/result.json" || ! -s "$data/manifest.json" \
          || ! -s "$data/synthetic_validation.json" \
          || ! -s "$data/oracle/sealed_oracle_control_summary.json" \
          || ! -s "$data/oracle/production_cache/manifest.json" ]]; then
        echo "status=invalid_complete_artifacts_incomplete"
        exit 65
    fi
    if ! env PYTHONPATH="$repo/src" "$python" - "$data" <<'PY'
from pathlib import Path
import sys

from cf4_aggregate_evidence_smc_execution import validate_published_bundle

value = validate_published_bundle(Path(sys.argv[1]))
if value["valid_scientific_complete"] is not True:
    raise SystemExit("invalid production completion")
PY
    then
        echo "status=invalid_complete_postcheck"
        exit 65
    fi
    cat "$state/COMPLETE"
elif [[ -s "$state/FAILED" ]]; then
    cat "$state/FAILED"
    [[ -s "$state/run.log" ]] && tail -n 20 "$state/run.log"
    exit 1
else
    cat "$state/RUNNING"
    [[ -s "$state/run.log" ]] && tail -n 12 "$state/run.log"
    exit 0
fi
