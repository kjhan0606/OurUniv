#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v53_v50_v52_domainwise_audit_program.json
expected_program_sha=d93dff7a1fff8ad49e9841b62e99d647bf2220b39ea9c165b0168e2ae5cae004
sequence=$tng/evaluation/tng100_simba_swift_v53_v50_v52_domainwise_audit_sequence
output=$tng/evaluation/tng100_simba_swift_v53_v50_v52_domainwise_audit/audit.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V53 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V53 requires clean worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$expected_program_sha" ]] || {
  echo "V53 frozen program hash differs" >&2
  exit 1
}
for path in "$sequence" "$(dirname "$output")"; do
  [[ ! -e $path ]] || {
    echo "V53 refuses existing output: $path" >&2
    exit 1
  }
done

mkdir -p "$sequence"
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" auditing >"$status"
python -u src/hong2021_v53_domainwise_audit.py \
  --program "$program" --repo "$repo" --out "$output" \
  >"$sequence/audit.log" 2>&1
