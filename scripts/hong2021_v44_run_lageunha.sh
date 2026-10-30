#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v44_local_mixture_copula_program.json
derived=$tng/derived/hong2021_v44
cache=$derived/conditioning_cache.h5
cache_report=$derived/conditioning_cache.json
checkpoint=$derived/local_mixture_copula_step12000.pt
report=$derived/local_mixture_copula_step12000.json
sequence=$tng/evaluation/tng100_simba_swift_v44_local_mixture_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v44_e24_local_mixture_copula

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V44 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V44 requires clean worktree" >&2
  exit 1
}
for path in "$derived" "$sequence" "$evaluation"; do
  [[ ! -e $path ]] || {
    echo "V44 refuses existing output: $path" >&2
    exit 1
  }
done

mkdir -p "$derived" "$sequence"
status=$sequence/status
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" complete >"$status"; else printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" preparing >"$status"
python -u src/hong2021_v44_train.py prepare \
  --program "$program" --repo "$repo" --out "$cache" --report "$cache_report" \
  >"$sequence/prepare.log" 2>&1
cache_sha=$(sha256sum "$cache" | awk '{print $1}')

printf "%s\n" preflight >"$status"
python -u src/hong2021_v44_train.py preflight \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$cache_sha" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf "%s\n" training >"$status"
python -u src/hong2021_v44_train.py train \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$cache_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --report "$report" >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')

printf "%s\n" sampling >"$status"
python -u src/hong2021_v44_sample.py \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$cache_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$evaluation" >"$sequence/sample.log" 2>&1

printf "%s\n" evaluating >"$status"
for arm in query_local_mixture_copula rolled_parameter_control structure_risk_ablation; do
  for domain in tng simba_dev swift_dev; do
    root=$evaluation/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" \
      --voxel-mpc-h .3125 >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" gating >"$status"
python -u src/hong2021_v44_development_gate.py \
  --root "$evaluation" --program "$program" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
