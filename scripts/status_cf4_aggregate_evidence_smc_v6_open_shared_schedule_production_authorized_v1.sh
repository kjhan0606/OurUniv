#!/usr/bin/env bash
set -Eeuo pipefail

readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run
readonly receipts=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_receipts
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python

# Read-only status based exclusively on lifecycle markers.
if [[ ! -e "$state" && ! -L "$state" ]]; then
    if [[ -L "$receipts" ]]; then
        echo 'status=invalid_dangling_receipt_root'
        exit 65
    fi
    if [[ -e "$receipts" ]]; then
        env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    _read_only_failed_status,
)
print("status=" + _read_only_failed_status(allow_state_absent=True)["status"])
PY
        exit 0
    fi
    echo 'status=absent'; exit 0
fi
if [[ ! -d "$state" || -L "$state" ]]; then
    echo 'status=invalid_state_type'
    exit 65
fi

markers=0
selected=
for name in RUNNING COMPLETE FAILED; do
    marker="$state/$name"
    if [[ -e "$marker" ]]; then
        ((markers += 1))
        selected=$name
        if [[ ! -f "$marker" || -L "$marker" || ! -s "$marker" ]]; then
            echo 'status=invalid_marker_type_or_empty'
            exit 65
        fi
        mode=$(stat -c '%a' "$marker")
        if [[ "$name" == RUNNING && "$mode" != 444 ]] \
                || [[ "$name" != RUNNING && "$mode" != 444 ]]; then
            echo 'status=invalid_marker_mode'
            exit 65
        fi
    fi
done
if (( markers == 0 )); then
    echo 'status=invalid_state_no_marker'
    exit 65
fi
if (( markers != 1 )); then
    echo 'status=invalid_state_conflicting_markers'
    exit 65
fi
case "$selected" in
    RUNNING)
        env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    _read_only_running_status,
)
print("status=" + _read_only_running_status()["status"])
PY
        ;;
    COMPLETE)
        env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    _read_only_complete_status,
)
value = _read_only_complete_status()
print("status=" + value["status"])
print("science_status=" + value["science_status"])
print("manifest_sha256=" + value["manifest_sha256"])
PY
        ;;
    FAILED)
        env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    _read_only_failed_status,
)
print("status=" + _read_only_failed_status()["status"])
PY
        ;;
    *) echo 'status=invalid_state'; exit 65 ;;
esac
