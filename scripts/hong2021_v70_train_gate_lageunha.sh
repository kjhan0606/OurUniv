#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_train_joint_structure_gate_program.json
program_sha=13ce1abfabe92a38072077637ef4f724f1951b6da6bd8214f472fd930c91f728
training=$tng/training/tng100_simba_swift_v70_latent_spatial
checkpoint=$training/step30000.pt
report=$training/training_report.json
root=$tng/evaluation/tng100_simba_swift_v70_train_joint_structure_gate
decision=$root/decision.json
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
for path in "$checkpoint" "$report"; do [[ -f $path ]] || exit 1; done
[[ ! -e $root ]] || exit 1
python - "$report" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x["status"] == "complete_fixed_30000_step_fit"
assert x["training_complete"] is True
assert x["train_only_mechanism_gate_run"] is False
assert x["validation_accessed"] is False
assert x["development_accessed"] is False
assert x["independent_gate_locked"] is True
PY
mkdir "$root"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V70_train_only_gate exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing_before_V70_train_only_gate >"$status"
pytest -q >"$sequence/train_gate_pytest.log" 2>&1
printf "%s\n" evaluating_V70_train_only_joint_structure_gate >"$status"
python -u src/hong2021_v70_train_gate.py \
  --program "$program" --repo "$repo" --out "$decision" >"$sequence/train_gate.log" 2>&1
selected=$(python - "$decision" <<'PY'
import json, sys
print("true" if json.load(open(sys.argv[1]))["candidate_selected"] else "false")
PY
)
if [[ $selected == true ]]; then
  printf "%s\n" complete_V70_train_only_gate_pass_locked_development_authorized >"$status"
else
  printf "%s\n" complete_V70_train_only_gate_rejection_development_locked >"$status"
fi
trap - EXIT
