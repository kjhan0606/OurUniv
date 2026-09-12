#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

output_dir=$1
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output path: $output_dir" >&2
  exit 73
fi

output_parent=$(dirname -- "$output_dir")
mkdir -p -- "$output_parent"
stage_dir=$(mktemp -d "${output_parent}/.2mpp-v1.XXXXXX")
cleanup() {
  if [[ -n "${stage_dir:-}" && -d "$stage_dir" ]]; then
    rm -rf -- "$stage_dir"
  fi
}
trap cleanup EXIT

tap_url="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
readme_url="https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/416/2840/ReadMe"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
validator="${repo_root}/src/cf4_2mpp_validate.py"

download_tap_table() {
  local table_name=$1
  local destination=$2
  curl \
    --fail \
    --show-error \
    --silent \
    --location \
    --connect-timeout 20 \
    --max-time 300 \
    --retry 3 \
    --retry-all-errors \
    --get "$tap_url" \
    --data-urlencode "REQUEST=doQuery" \
    --data-urlencode "LANG=ADQL" \
    --data-urlencode "FORMAT=csv" \
    --data-urlencode "MAXREC=100000" \
    --data-urlencode "QUERY=SELECT * FROM \"J/MNRAS/416/2840/${table_name}\" ORDER BY recno" \
    --output "$destination"
}

download_tap_table "catalog" "${stage_dir}/2mpp_catalog.csv"
download_tap_table "group" "${stage_dir}/2mpp_groups.csv"
curl \
  --fail \
  --show-error \
  --silent \
  --location \
  --connect-timeout 20 \
  --max-time 120 \
  --retry 3 \
  --retry-all-errors \
  "$readme_url" \
  --output "${stage_dir}/2mpp_ReadMe.txt"

python3 "$validator" \
  --catalog "${stage_dir}/2mpp_catalog.csv" \
  --groups "${stage_dir}/2mpp_groups.csv" \
  --readme "${stage_dir}/2mpp_ReadMe.txt"

mv -- "$stage_dir" "$output_dir"
stage_dir=""
trap - EXIT
