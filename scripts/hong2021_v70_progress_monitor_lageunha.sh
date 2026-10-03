#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
training=$tng/training/tng100_simba_swift_v70_latent_spatial
train_gate=$tng/evaluation/tng100_simba_swift_v70_train_joint_structure_gate/decision.json
development=$tng/evaluation/tng100_simba_swift_v70_development/development_decision.json
sealed=$sequence/sealed_result.json
status_path=$sequence/status
history=$training/history.json
progress=$sequence/progress.json

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
mkdir -p "$sequence"

while true; do
  status=$(cat "$status_path" 2>/dev/null || true)
  training_alive=false
  supervisor_alive=false
  tmux has-session -t hong_v70_train 2>/dev/null && training_alive=true
  tmux has-session -t hong_v70_supervisor 2>/dev/null && supervisor_alive=true
  python - "$repo" "$history" "$train_gate" "$development" "$sealed" \
    "$status" "$training_alive" "$supervisor_alive" "$progress" <<'PY'
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

(
    repo, history_path, train_gate_path, development_path, sealed_path,
    status, training_alive, supervisor_alive, output_path,
) = sys.argv[1:]
repo = Path(repo)


def artifact(value: str) -> dict:
    path = Path(value)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }


history = []
if Path(history_path).is_file():
    history = json.loads(Path(history_path).read_text())
latest = history[-1] if history else None
head = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
dirty = subprocess.run(
    ["git", "-C", str(repo), "status", "--porcelain"],
    check=True, capture_output=True, text=True,
).stdout.strip()
terminal_prefixes = (
    "complete_V70_train_only_gate_rejection_development_locked",
    "complete_V70_development_pass_waiting_explicit_EAGLE_approval",
    "complete_V70_development_failure_independent_gate_locked",
    "failed_V70_",
)
terminal = status.startswith(terminal_prefixes)
result = {
    "schema": "hong2021-v70-live-progress-v1",
    "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "status": status,
    "terminal": terminal,
    "repo_head": head,
    "worktree_clean": not bool(dirty),
    "tmux": {
        "training_alive": training_alive == "true",
        "supervisor_alive": supervisor_alive == "true",
    },
    "training": {
        "history_rows": len(history),
        "latest": latest,
    },
    "artifacts": {
        "train_gate": artifact(train_gate_path),
        "development_decision": artifact(development_path),
        "sealed_result": artifact(sealed_path),
    },
    "firewall": {
        "validation_or_development_payload_read": False,
        "EAGLE_accessed": False,
        "training_or_gate_modified": False,
    },
}
output = Path(output_path)
partial = output.with_suffix(output.suffix + ".partial")
partial.write_text(json.dumps(result, indent=2) + "\n")
os.replace(partial, output)
print(result["updated_at"], status, latest["step"] if latest else None, flush=True)
PY
  case "$status" in
    complete_V70_train_only_gate_rejection_development_locked*|\
    complete_V70_development_pass_waiting_explicit_EAGLE_approval*|\
    complete_V70_development_failure_independent_gate_locked*|\
    failed_V70_*)
      exit 0
      ;;
  esac
  sleep 60
done
