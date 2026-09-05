#!/usr/bin/env bash
set -Eeuo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open.json"
readonly receipt_root=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts
readonly pilot=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run
host=$(hostname); readonly host
short=${host%%.*}; readonly short
lower=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$short"); readonly lower
if [[ "$lower" != "$expected_host" ]]; then echo "host gate failed: $host" >&2; exit 69; fi
# Read-only gate: it exits before receipt, namespace, or output creation.
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6_open import PROGRAM,load_program,require_execution_authorization
if Path(sys.argv[1]).resolve()!=PROGRAM.resolve(): raise SystemExit('noncanonical program')
require_execution_authorization(load_program(),'pilot')
PY
exit 65
