#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_latent_spatial_score_model_program.json
program_sha=79f1b5fe1462664b9b7a237bd82a821e205f3901603d64801a01b328c43f7e42
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
preflight=$sequence/preflight.json
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ ! -e $sequence ]] || exit 1
mkdir -p "$sequence"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V70_preflight exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1
printf "%s\n" preflighting_latent_spatial_model >"$status"
python -u src/hong2021_v70_preflight.py \
  --program "$program" --repo "$repo" --out "$preflight" >"$sequence/preflight.log" 2>&1
printf "%s\n" complete_V70_hard_preflight_pass >"$status"
trap - EXIT
