#!/usr/bin/env bash
set -euo pipefail

# Resume only the exact V20 gate whose first sampler was blocked by the
# pre-scientific edm_p_mean_mode validator literal.  No model, checkpoint,
# sample seed, candidate, evaluator, or gate setting is changed.
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v20_e8_gaussianized_marginal_edm
evaluation=$tng/evaluation/tng100_simba_swift_v20_e8_gaussianized_marginal
archive=$tng/evaluation/tng100_simba_swift_v20_e8_gaussianized_marginal.infrastructure_failure_validator_literal
sequence=$tng/evaluation/tng100_simba_swift_v20_sequence
status=$sequence/status.json
sample_log=$evaluation/development_candidates/step_005000/tng/sample.log
cd "$repo"

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v20-development-supervisor-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "independent_data_paths_accessed": False,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V20-E8 gate resume must run on Lageunha, not $host" >&2
  exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "V20-E8 gate resume requires a clean committed worktree" >&2
  exit 1
fi
if [[ ! -d $evaluation ]]; then
  echo "V20 failed evaluation directory is missing" >&2
  exit 1
fi
python - "$status" "$training/run.json" "$sample_log" <<'PY'
import json, sys
from pathlib import Path
status, run, log = map(Path, sys.argv[1:])
s = json.loads(status.read_text())
if s.get("state") != "failed" or "hong2021_v20_gate.sh" not in s.get("detail", ""):
    raise RuntimeError("V20 resume did not find the exact failed gate state")
if s.get("independent_data_paths_accessed") is not False:
    raise RuntimeError("independent-data firewall was not preserved")
if json.loads(run.read_text()).get("status") != "complete":
    raise RuntimeError("V20 training is not complete")
expected = "V20 checkpoint did not use the E3 relative-noise mode"
if expected not in log.read_text():
    raise RuntimeError("V20 gate did not fail at the audited validator literal")
PY
if [[ -e $archive ]]; then
  echo "V20 validator-failure archive already exists" >&2
  exit 1
fi
if find "$evaluation" -type f ! -path "$sample_log" -print -quit | grep -q .; then
  echo "V20 failed gate contains an unexpected artifact" >&2
  exit 1
fi
failure() {
  local code=$?
  trap - ERR
  write_status failed_gate_resume "exit=$code; exact scientific configuration unchanged"
  exit "$code"
}
trap failure ERR
mv "$evaluation" "$archive"
write_status running_predeclared_e8_sampling_gate_after_validator_literal_fix "$evaluation"
bash scripts/hong2021_v20_gate.sh >"$sequence/gate_resume.log" 2>&1

decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
next=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["next"])' "$decision")
trap - ERR
if [[ $passed == True ]]; then
  write_status complete_e8_passed_astrid_still_unopened "$decision"
  exec scripts/hong2021_v20_finalize.sh
else
  write_status complete_e8_failed_astrid_unopened "$next"
fi
