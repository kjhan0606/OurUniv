#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program=$repo/config/hong2021_v62_conditional_moment_gradient_audit_program.json
root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v62_conditional_moment_gradient_audit
out=$root/audit.json
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v62_conditional_moment_gradient_audit_sequence

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != lageunha ]]; then
  echo "V62 requires Lageunha" >&2
  exit 1
fi
cd "$repo"
if [[ -n $(git status --porcelain) ]]; then
  echo "V62 requires a clean worktree" >&2
  exit 1
fi
if [[ -e "$out" ]]; then
  echo "V62 refuses existing output" >&2
  exit 1
fi
mkdir -p "$root" "$sequence"
PYTHONPATH=src pytest -q | tee "$sequence/pytest.log"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 taskset -c 100-115 python -u \
  src/hong2021_v62_conditional_moment_gradient_audit.py \
  --program "$program" \
  --repo "$repo" \
  --out "$out" | tee "$sequence/audit.log"
jq -e '.status == "complete_no_refit_train_only_objective_gradient_audit" and .training_or_refit_performed == false and .development_accessed == false and .independent_gate_locked == true' "$out" >/dev/null
echo complete > "$sequence/status"
