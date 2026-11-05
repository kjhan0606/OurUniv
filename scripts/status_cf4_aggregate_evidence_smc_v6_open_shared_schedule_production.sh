#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly state=${CF4_V6_SHARED_STATUS_STATE:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run}
readonly data=${CF4_V6_SHARED_STATUS_DATA:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1}
readonly cache=${CF4_V6_SHARED_STATUS_CACHE:-/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache}

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
    if [[ ! -d "$cache" \
          || "$(stat -c %a "$state")" != 555 \
          || "$(stat -c %a "$data")" != 555 \
          || "$(stat -c %a "$cache")" != 555 \
          || "$(stat -c %a "$state/COMPLETE")" != 444 ]]; then
        echo "status=invalid_complete_mode_contract"
        exit 65
    fi
    if [[ ! -s "$data/result.json" || ! -s "$data/manifest.json" ]]; then
        echo "status=invalid_complete_artifacts_incomplete"
        exit 65
    fi
    if ! env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
        PYTHONPATH="$repo/src" "$python" - "$data" <<'PY'
from pathlib import Path
import sys

from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution import (
    validate_published_bundle,
)

value = validate_published_bundle(Path(sys.argv[1]))
if value.get("valid_scientific_complete") is not True:
    raise SystemExit("invalid production completion")
PY
    then
        echo "status=invalid_complete_postcheck"
        exit 65
    fi
    cat "$state/COMPLETE"
elif [[ -s "$state/FAILED" ]]; then
    if [[ "$(stat -c %a "$state/FAILED")" != 444 ]]; then
        echo "status=invalid_failed_marker_mode"
        exit 65
    fi
    cat "$state/FAILED"
    exit 1
else
    cat "$state/RUNNING"
    exit 0
fi
