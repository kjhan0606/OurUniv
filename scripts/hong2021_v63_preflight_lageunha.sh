#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program=$repo/config/hong2021_v63_conditional_moment_model_program.json
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v63_conditional_moment_sequence
out=$sequence/preflight.json

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != lageunha ]]; then
  echo "V63 preflight requires Lageunha" >&2
  exit 1
fi
cd "$repo"
if [[ -n $(git status --porcelain) ]]; then
  echo "V63 preflight requires a clean worktree" >&2
  exit 1
fi
if [[ -e "$out" ]]; then
  echo "V63 refuses existing preflight" >&2
  exit 1
fi
mkdir -p "$sequence"
PYTHONPATH=src pytest -q | tee "$sequence/pytest.log"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 taskset -c 100-115 python -u \
  src/hong2021_v63_preflight.py \
  --program "$program" \
  --repo "$repo" \
  --out "$out" | tee "$sequence/preflight.log"
if jq -e '.status == "pass" and .training_performed == false and .validation_accessed == false and .development_accessed == false and .independent_gate_locked == true' "$out" >/dev/null; then
  echo complete_preflight_pass > "$sequence/status"
else
  echo complete_preflight_failure_training_locked > "$sequence/status"
  exit 1
fi
