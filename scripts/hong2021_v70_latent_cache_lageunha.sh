#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_latent_spatial_score_model_program.json
program_sha=79f1b5fe1462664b9b7a237bd82a821e205f3901603d64801a01b328c43f7e42
record=$repo/config/hong2021_v70_preflight_result_record.json
record_sha=2049b216d00e734c6c8ffa45966112880ff01bfdd191754c7fb6a6c22e693050
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
preflight=$sequence/preflight.json
preflight_sha=5b708473534954ff45f19ae0711249dd2d7305fa7288458467b71a78b853a3c4
derived=$tng/derived/hong2021_v70
cache=$derived/train_latent.h5
report=$derived/train_latent.json
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(sha256sum "$record" | awk '{print $1}') == "$record_sha" ]] || exit 1
[[ $(sha256sum "$preflight" | awk '{print $1}') == "$preflight_sha" ]] || exit 1
for path in "$cache" "$report"; do [[ ! -e $path ]] || exit 1; done
mkdir -p "$derived"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V70_latent_cache exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing_before_latent_cache >"$status"
pytest -q >"$sequence/cache_pytest.log" 2>&1
printf "%s\n" building_and_scanning_train_only_latent_cache >"$status"
python -u src/hong2021_v70_latent_cache.py \
  --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$cache" --report "$report" >"$sequence/cache.log" 2>&1
printf "%s\n" complete_V70_latent_cache_scan_pass >"$status"
trap - EXIT
