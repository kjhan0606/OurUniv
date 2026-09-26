#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v43_tail_threshold_target_audit_program.json
root=$tng/evaluation/tng100_simba_swift_v43_tail_threshold_target_audit
audit=$root/audit.json

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V43 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V43 requires clean worktree" >&2
  exit 1
}
[[ ! -e $root ]] || {
  echo "V43 refuses existing output: $root" >&2
  exit 1
}

mkdir -p "$root"
status=$root/status
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$root/pytest.log" 2>&1
printf "%s\n" auditing >"$status"
python -u src/hong2021_v43_tail_target_audit.py \
  --program "$program" --repo "$repo" --out "$audit" \
  >"$root/audit.log" 2>&1
