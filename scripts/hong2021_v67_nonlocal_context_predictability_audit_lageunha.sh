#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v67_nonlocal_context_predictability_audit_program.json
program_sha=e200bb88c3e820c0350067db926f724b28b3cdef22c58bf6dd3ad07e6070933a
root=$tng/evaluation/tng100_simba_swift_v67_nonlocal_context_predictability_audit
sequence=$tng/evaluation/tng100_simba_swift_v67_nonlocal_context_predictability_sequence
audit=$root/audit.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V67 audit requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V67 audit requires clean worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V67 frozen program hash differs" >&2
  exit 1
}
for path in "$root" "$sequence"; do
  [[ ! -e $path ]] || {
    echo "V67 refuses existing output: $path" >&2
    exit 1
  }
done
mkdir -p "$sequence"

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V67_audit exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" auditing_train_only_target_free_nonlocal_context >"$status"
python -u src/hong2021_v67_nonlocal_context_predictability_audit.py \
  --program "$program" --repo "$repo" --out "$audit" \
  >"$sequence/audit.log" 2>&1
printf "%s\n" complete_train_only_target_free_nonlocal_context_predictability_audit >"$status"
trap - EXIT
