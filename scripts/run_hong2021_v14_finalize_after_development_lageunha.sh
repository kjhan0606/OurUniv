#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
decision=$tng/evaluation/tng100_simba_swift_v14_multiscale_edm/development_decision.json
sequence=$tng/evaluation/tng100_simba_swift_v14_multiscale_edm/sequence_status.json
finalize_root=$tng/evaluation/tng100_simba_swift_v14_finalize
status=$finalize_root/status.json
seal=config/hong2021_v14_astrid_one_shot_seal.json
session=hong_v14_astrid_one_shot
mkdir -p "$finalize_root"
cd "$repo"
export PYTHONPATH=$repo/src

write_status() {
    python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v14-finalize-after-development-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_three_domain_development_gate "$sequence"
while [[ ! -s $decision ]]; do
    if [[ -s $sequence ]]; then
        stage=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$sequence")
        if [[ $stage == complete_failed_development_no_astrid ]]; then
            write_status complete_development_failed_astrid_unopened "$sequence"
            exit 0
        fi
    fi
    sleep 30
done
passed=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
if [[ $passed != True ]]; then
    write_status complete_development_failed_astrid_unopened "$decision"
    exit 0
fi

write_status creating_exact_artifact_seal "$decision"
if [[ -e $seal ]]; then
    printf 'Seal already exists before the dedicated seal commit: %s\n' "$seal" >&2
    exit 2
fi
if [[ -n $(git status --porcelain) ]]; then
    printf 'Repository became dirty before V14 sealing.\n' >&2
    git status --short >&2
    exit 2
fi
python src/hong2021_v14_freeze.py create --repo "$repo" --tng "$tng" \
    --astrid-root /gpfs/kjhan/CAMELS/Astrid/L25n256 --out "$seal" \
    >"$finalize_root/create_seal.log" 2>&1
git add -- "$seal"
staged=$(git diff --cached --name-only)
if [[ $staged != "$seal" ]]; then
    printf 'Seal commit contains an unexpected path: %s\n' "$staged" >&2
    exit 2
fi
git commit -m "Freeze V14 Astrid one-shot artifacts"
git push

write_status verifying_committed_seal "$seal"
python src/hong2021_v14_freeze.py verify --repo "$repo" --seal "$seal" \
    --require-unopened >"$finalize_root/verify_seal.log" 2>&1

write_status launching_astrid_one_shot "$session"
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Astrid supervisor session already exists: %s\n' "$session" >&2
    exit 2
fi
tmux new-session -d -s "$session" \
    "cd '$repo' && scripts/supervise_hong2021_v14_astrid_one_shot_lageunha.sh '$seal' >> '$finalize_root/astrid_supervisor.log' 2>&1"
write_status complete_astrid_supervisor_launched "$session"
