#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v20_sequence/status.json
finalize_root=$tng/evaluation/tng100_simba_swift_v20_finalize
status=$finalize_root/status.json
seal=config/hong2021_v20_astrid_one_shot_seal.json
astrid_session=hong2021_v20_astrid
mkdir -p "$finalize_root"
cd "$repo"

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v20-finalize-after-development-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_v20_development_sequence "$sequence"
while true; do
  if [[ -s $sequence ]]; then
    state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$sequence")
    case "$state" in
      complete_e8_passed_astrid_still_unopened)
        write_status creating_committed_v20_seal "$sequence"
        if [[ -n $(git status --porcelain) ]]; then
          write_status stopped_dirty_worktree_before_seal "$(git status --short)"
          exit 1
        fi
        python src/hong2021_v20_freeze.py create \
          --repo "$repo" --tng "$tng" --out "$seal" \
          >"$finalize_root/seal_creation.log" 2>&1
        git add -- "$seal"
        git commit -m "Seal V20 Astrid one-shot"
        git push origin HEAD
        python src/hong2021_v20_freeze.py verify \
          --repo "$repo" --seal "$seal" --require-unopened \
          >"$finalize_root/seal_verification.log" 2>&1
        if tmux has-session -t "$astrid_session" 2>/dev/null; then
          write_status stopped_existing_astrid_tmux_session "$astrid_session"
          exit 1
        fi
        tmux new-session -d -s "$astrid_session" \
          "cd '$repo' && exec scripts/supervise_hong2021_v20_astrid_one_shot_lageunha.sh '$seal' >>'$finalize_root/astrid_supervisor.log' 2>&1"
        write_status complete_sealed_pushed_and_astrid_supervisor_launched \
          "$astrid_session"
        exit 0 ;;
      complete_e8_failed_astrid_unopened)
        write_status complete_development_failed_astrid_unopened "$sequence"
        exit 0 ;;
      failed*)
        write_status stopped_development_infrastructure_failure "$sequence"
        exit 1 ;;
    esac
  fi
  sleep 30
done
