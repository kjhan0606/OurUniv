#!/usr/bin/env bash
set -Eeuo pipefail
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_lageunha.sh
[[ -x "$runner" ]] || { printf 'canonical pilot runner is unavailable\n' >&2; exit 65; }
exec "$runner"
