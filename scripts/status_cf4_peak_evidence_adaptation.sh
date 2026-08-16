#!/usr/bin/env bash
set -Eeuo pipefail

readonly state=/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_adaptation_v1

if [[ -s "$state/COMPLETE" ]]; then
    cat "$state/COMPLETE"
elif [[ -s "$state/FAILED" ]]; then
    cat "$state/FAILED"
    [[ -s "$state/run.log" ]] && tail -n 20 "$state/run.log"
elif [[ -s "$state/RUNNING" ]]; then
    cat "$state/RUNNING"
    [[ -s "$state/run.log" ]] && tail -n 8 "$state/run.log"
else
    echo "status=not_started"
fi
