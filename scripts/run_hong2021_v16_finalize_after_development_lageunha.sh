#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v16_sequence/status.json
finalize_root=$tng/evaluation/tng100_simba_swift_v16_finalize
status=$finalize_root/status.json
seal=config/hong2021_v16_astrid_one_shot_seal.json
session=hong_v16_astrid_one_shot
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
    "schema": "hong2021-v16-finalize-after-development-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_v16_development_sequence "$sequence"
while true; do
  if [[ -s $sequence ]]; then
    state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$sequence")
    case "$state" in
      complete_e4_passed_astrid_still_unopened)
        break
        ;;
      complete_e4_failed_astrid_unopened)
        write_status complete_development_failed_astrid_unopened "$sequence"
        exit 0
        ;;
      failed*)
        write_status stopped_development_infrastructure_failure "$sequence"
        exit 1
        ;;
    esac
  fi
  sleep 30
done

write_status creating_exact_v16_artifact_seal "$sequence"
if [[ -e $seal ]]; then
  echo "V16 seal already exists before its dedicated seal commit: $seal" >&2
  exit 2
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "Repository became dirty before V16 sealing" >&2
  git status --short >&2
  exit 2
fi
python src/hong2021_v16_freeze.py create --repo "$repo" --tng "$tng" \
  --astrid-root /gpfs/kjhan/CAMELS/Astrid/L25n256 --out "$seal" \
  >"$finalize_root/create_seal.log" 2>&1
git add -- "$seal"
staged=$(git diff --cached --name-only)
if [[ $staged != "$seal" ]]; then
  echo "V16 seal commit contains unexpected paths: $staged" >&2
  exit 2
fi
git commit -m "Freeze V16 Astrid one-shot artifacts"
git push

write_status verifying_committed_v16_seal "$seal"
python src/hong2021_v16_freeze.py verify --repo "$repo" --seal "$seal" \
  --require-unopened >"$finalize_root/verify_seal.log" 2>&1
write_status launching_v16_astrid_one_shot "$session"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "V16 Astrid supervisor already exists: $session" >&2
  exit 2
fi
tmux new-session -d -s "$session" \
  "cd '$repo' && scripts/supervise_hong2021_v16_astrid_one_shot_lageunha.sh '$seal' >> '$finalize_root/astrid_supervisor.log' 2>&1"
write_status complete_astrid_supervisor_launched "$session"
