#!/usr/bin/env bash
# v6 one-shot runner. It ships execution-false and performs no reservation.
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6.json"
readonly grant="$repo/config/cf4_aggregate_evidence_smc_execution_grant_v6.json"
readonly release=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_release.json
readonly manifest=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_manifest.json
readonly receipt_root=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_receipts
readonly pilot_root=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_disposable_pilot
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_run

host=$(hostname); readonly host
host_short=${host%%.*}; readonly host_short
host_short_ascii_lower=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$host_short")
readonly host_short_ascii_lower
if [[ "$host_short_ascii_lower" != "$expected_host" ]]; then
    echo "host gate failed: $host" >&2; exit 69
fi

# This read-only authorization call precedes every namespace, lock, output,
# process, or resource operation. With no future grant it always exits here.
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6 import (
    CANONICAL_PROGRAM, load_canonical_authorization_program,
    require_execution_authorization,
)
if Path(sys.argv[1]).resolve() != CANONICAL_PROGRAM.resolve():
    raise SystemExit("v6 program path is not canonical")
require_execution_authorization(load_canonical_authorization_program())
PY

# Reaching this line requires a future separately implemented runner.
echo "unreachable v6 execution path" >&2
exit 65
