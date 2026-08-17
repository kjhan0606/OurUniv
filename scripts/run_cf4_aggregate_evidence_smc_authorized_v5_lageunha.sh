#!/usr/bin/env bash
# v5 one-shot runner.  Shipped execution-false: no grant/release/manifest exists.
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly expected_host=lageunha
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v5.json"
readonly grant="$repo/config/cf4_aggregate_evidence_smc_execution_grant_v5.json"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_run
readonly receipts=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_receipts
readonly receipt="$receipts/one-shot-receipt"
readonly running="$state/RUNNING"
readonly complete="$state/COMPLETE"
readonly failed="$state/FAILED"

finish() {
    local rc=$?
    if (( rc == 0 )) && [[ "${validated_complete:-false}" == true ]]; then
        {
            printf 'status=complete\nscience_status=%s\noutcome_kind=%s\nfailure_class=%s\n' \
                "$science_status" "$outcome_kind" "$failure_class"
            printf 'preflight_snapshot_sha256=%s\nresult=%s\nresult_sha256=%s\n' \
                "$snapshot_sha" "$result" "$result_sha"
            printf 'manifest=%s\nmanifest_sha256=%s\n' "$manifest" "$manifest_sha"
            printf 'automatic_retry_retune_scale_up_or_follow_on=false\n'
        } >"$complete"
    else
        printf 'status=failed\nexit_code=%s\nfailure_class=invalid_provenance_or_execution\n' "$rc" >"$failed"
    fi
    rm -f "$running"
}

host=$(hostname); readonly host
if [[ "${host%%.*}" != "$expected_host" ]]; then
    echo "host gate failed" >&2; exit 69
fi
if [[ -e "$data" || -e "$state" || -e "$receipt" ]]; then
    echo "v5 namespace or exclusive receipt already exists" >&2; exit 73
fi

# This is read-only until the separately issued grant and paired external
# release/manifest prove authorization.  It currently exits here.
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import (
    CANONICAL_PROGRAM, load_canonical_authorization_program,
    require_execution_authorization,
)
if Path(sys.argv[1]).resolve() != CANONICAL_PROGRAM.resolve():
    raise SystemExit("v5 program path is not canonical")
require_execution_authorization(load_canonical_authorization_program())
PY

mkdir "$receipts"
snapshot_sha=$(env PYTHONPATH="$repo/src" "$python" - "$receipt" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import (
    create_preflight_receipt, load_canonical_authorization_program,
)
_, value = create_preflight_receipt(Path(sys.argv[1]), load_canonical_authorization_program())
print(value)
PY
)
readonly snapshot_sha

mkdir "$state" "$data"
trap finish EXIT
printf 'status=running\nstage=aggregate_evidence_smc_authorized_v5\npreflight_snapshot_sha256=%s\nreceipt=%s\n' \
    "$snapshot_sha" "$receipt" >"$running"
printf 'preflight_snapshot_sha256=%s\nreceipt=%s\n' "$snapshot_sha" "$receipt" >"$state/environment.txt"

# Exact receipt/release revalidation happens immediately before core, after
# core, and again before COMPLETE.  The receipt's release.anchor hard link is
# required to name the same inode as the canonical release. Any missing anchor, changed inode, changed
# link count, content hash, ID, grant, program, implementation, or runner
# routes through FAILED via the EXIT trap.
env PYTHONPATH="$repo/src" "$python" - "$program" "$receipt" "$snapshot_sha" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import (
    load_canonical_authorization_program, revalidate_preflight_receipt,
    run_authorized_v5,
)
program = load_canonical_authorization_program()
revalidate_preflight_receipt(Path(sys.argv[2]), sys.argv[3], program)
run_authorized_v5(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
revalidate_preflight_receipt(Path(sys.argv[2]), sys.argv[3], program)
PY

# This pre-COMPLETE read-only check permits exactly the two scientific terminal
# statuses. A missing, writable, malformed, or invalid bundle exits nonzero
# and therefore reaches FAILED rather than masquerading as COMPLETE.
postcheck=$(env PYTHONPATH="$repo/src" "$python" - "$data" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import read_only_science_postcheck

checked = read_only_science_postcheck(Path(sys.argv[1]))
print("\t".join(checked[key] for key in (
    "science_status", "outcome_kind", "failure_class", "result",
    "result_sha256", "manifest", "manifest_sha256",
)))
PY
)
IFS=$'\t' read -r science_status outcome_kind failure_class result result_sha manifest manifest_sha <<<"$postcheck"
if [[ "$science_status" != complete_pass_production_smc \
      && "$science_status" != complete_scientific_fail_production_smc ]] \
      || [[ -z "$outcome_kind" || -z "$failure_class" || -z "$result_sha" || -z "$manifest_sha" ]]; then
    echo "read-only science postcheck did not return a valid terminal status" >&2
    exit 65
fi
readonly science_status outcome_kind failure_class result result_sha manifest manifest_sha

# Final exact receipt snapshot check is immediately before COMPLETE.
env PYTHONPATH="$repo/src" "$python" - "$receipt" "$snapshot_sha" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import (
    load_canonical_authorization_program, revalidate_preflight_receipt,
)
revalidate_preflight_receipt(
    Path(sys.argv[1]), sys.argv[2], load_canonical_authorization_program()
)
PY
validated_complete=true
