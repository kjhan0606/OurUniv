#!/usr/bin/env bash
set -Eeuo pipefail

readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_run
readonly receipts=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_receipts
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly module=cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1

if [[ ! -e "$state" && ! -L "$state" ]]; then
    if [[ -L "$receipts" ]]; then
        echo 'status=invalid_dangling_receipt_root'; exit 65
    fi
    if [[ -e "$receipts" ]]; then
        /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" -c \
            "from $module import _read_only_failed_status; print('status='+_read_only_failed_status(allow_state_absent=True)['status'])"
        exit 0
    fi
    echo 'status=absent'; exit 0
fi
if [[ ! -d "$state" || -L "$state" ]]; then
    echo 'status=invalid_state_type'; exit 65
fi

markers=0
selected=
for name in RUNNING COMPLETE FAILED; do
    marker="$state/$name"
    if [[ -e "$marker" || -L "$marker" ]]; then
        ((markers += 1))
        selected=$name
        if [[ ! -f "$marker" || -L "$marker" || ! -s "$marker" ]]; then
            echo 'status=invalid_marker_type_or_empty'; exit 65
        fi
        if [[ $(stat -c '%a' "$marker") != 444 ]]; then
            echo 'status=invalid_marker_mode'; exit 65
        fi
    fi
done
if (( markers == 0 )); then
    echo 'status=invalid_state_no_marker'; exit 65
fi
if (( markers != 1 )); then
    echo 'status=invalid_state_conflicting_markers'; exit 65
fi

case "$selected" in
    RUNNING)
        code="from $module import _read_only_running_status as f; print('status='+f()['status'])"
        ;;
    COMPLETE)
        code="from $module import _read_only_complete_status as f; v=f(); print('status='+v['status']); print('science_status='+v['science_status']); print('manifest_sha256='+v['manifest_sha256'])"
        ;;
    FAILED)
        code="from $module import _read_only_failed_status as f; print('status='+f()['status'])"
        ;;
    *) echo 'status=invalid_state'; exit 65 ;;
esac
/usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" -c "$code"
