#!/usr/bin/env bash
set -Eeuo pipefail
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run
if [[ -f "$state/COMPLETE" ]]; then cat "$state/COMPLETE"; exit 0; fi
if [[ -f "$state/FAILED" ]]; then cat "$state/FAILED"; exit 1; fi
printf 'status=not_started_fail_closed\nstate=%s\n' "$state"
exit 3
