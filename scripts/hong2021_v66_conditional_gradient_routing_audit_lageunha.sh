#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v66_conditional_gradient_routing_audit_program.json
program_sha=f3aaa8ee8852341675bcdec264ba991a4d23b4ed6550cfa309e1b85d5b7730ca
root=$tng/evaluation/tng100_simba_swift_v66_conditional_gradient_routing_audit
sequence=$tng/evaluation/tng100_simba_swift_v66_conditional_gradient_routing_sequence
audit=$root/audit.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V66 audit requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V66 audit requires clean worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V66 frozen program hash differs" >&2
  exit 1
}
for path in "$root" "$sequence"; do
  [[ ! -e $path ]] || {
    echo "V66 refuses existing output: $path" >&2
    exit 1
  }
done
mkdir -p "$sequence"

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V66_audit exit=%s previous=%s\n" "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" auditing_no_refit_train_only_conditional_gradient_routing >"$status"
python -u src/hong2021_v66_conditional_gradient_routing_audit.py \
  --program "$program" --repo "$repo" --out "$audit" \
  >"$sequence/audit.log" 2>&1
printf "%s\n" complete_no_refit_train_only_conditional_gradient_routing_audit >"$status"
trap - EXIT
