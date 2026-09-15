#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v51_bounded_support_calibration_audit_program.json
expected_program_sha=ba30af32cfd147e97ebd62118536cab5b0c9b30cdb81a17538fed5305f1928ba
root=$tng/evaluation/tng100_simba_swift_v51_bounded_support_calibration_audit
output=$root/audit.json
status=$root/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V51 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V51 requires clean worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$expected_program_sha" ]] || {
  echo "V51 frozen program hash differs" >&2
  exit 1
}
[[ ! -e $root ]] || {
  echo "V51 refuses existing output: $root" >&2
  exit 1
}

mkdir -p "$root"
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$root/pytest.log" 2>&1

printf "%s\n" auditing >"$status"
python -u src/hong2021_v51_bounded_support_audit.py \
  --program "$program" \
  --repo "$repo" \
  --out "$output" >"$root/audit.log" 2>&1
sha256sum "$output" >"$root/audit.sha256"
