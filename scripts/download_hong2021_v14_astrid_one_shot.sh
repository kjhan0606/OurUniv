#!/usr/bin/env bash
set -euo pipefail

# This downloader has no development-mode entry point.  The committed V14
# Astrid runner sets the guard only after verifying the exact artifact seal.
if [[ ${HONG2021_V14_ASTRID_ONE_SHOT:-} != sealed \
   && ${HONG2021_V15_ASTRID_ONE_SHOT:-} != sealed \
   && ${HONG2021_V16_ASTRID_ONE_SHOT:-} != sealed ]]; then
    printf 'Refusing Astrid download outside the sealed one-shot runner.\n' >&2
    exit 2
fi

seal=${1:?usage: download_hong2021_v14_astrid_one_shot.sh SEAL [ROOT]}
root=${2:-/gpfs/kjhan/CAMELS/Astrid/L25n256}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
public=https://users.flatironinstitute.org/~camels
workers=${CAMELS_DOWNLOAD_WORKERS:-3}
if [[ ! $workers =~ ^[1-9][0-9]*$ ]]; then
    printf 'CAMELS_DOWNLOAD_WORKERS must be a positive integer\n' >&2
    exit 2
fi

cd "$repo"
export PYTHONPATH=$repo/src
python - "$seal" "$repo" <<'PY'
import sys
from hong2021_astrid_seal import verify_astrid_seal
verify_astrid_seal(sys.argv[1], repo=sys.argv[2], require_committed=True)
PY
mkdir -p "$root/raw" "$root/CV" "$root/download"

remote_bytes() {
    local url=$1 value
    value=$(curl --fail --location --silent --show-error --head "$url" \
        | awk 'BEGIN{IGNORECASE=1} /^content-length:/{gsub("\r",""); n=$2} END{print n}')
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        printf 'Cannot determine Content-Length for %s\n' "$url" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

download_one() {
    local url=$1 destination=$2 expected current partial
    expected=$(remote_bytes "$url")
    partial=$destination.partial
    mkdir -p "$(dirname "$destination")"
    if [[ -e $destination ]]; then
        current=$(stat -c %s "$destination")
        if [[ $current -ne $expected ]]; then
            printf 'Existing file has wrong size: %s (%s != %s)\n' \
                "$destination" "$current" "$expected" >&2
            return 1
        fi
        printf '[existing] %s %s\n' "$expected" "$destination"
        return 0
    fi
    if [[ -e $partial ]]; then
        current=$(stat -c %s "$partial")
        if [[ $current -gt $expected ]]; then
            printf 'Partial file exceeds remote size: %s\n' "$partial" >&2
            return 1
        fi
    fi
    curl --fail --location --retry 20 --retry-delay 5 --continue-at - \
        --silent --show-error --output "$partial" "$url"
    current=$(stat -c %s "$partial")
    if [[ $current -ne $expected ]]; then
        printf 'Incomplete download: %s (%s != %s)\n' \
            "$partial" "$current" "$expected" >&2
        return 1
    fi
    mv "$partial" "$destination"
    printf '[complete] %s %s\n' "$expected" "$destination"
}

download_realization() {
    local realization=$1
    local snapshot_url catalog_url snapshot catalog
    snapshot_url=$public/Sims/Astrid/L25n256/CV/CV_${realization}/snapshot_090.hdf5
    catalog_url=$public/FOF_Subfind/Astrid/L25n256/CV/CV_${realization}/groups_090.hdf5
    snapshot=$root/raw/CV_${realization}/snapshot_090.hdf5
    catalog=$root/CV/CV_${realization}/groups_090.hdf5
    download_one "$catalog_url" "$catalog"
    download_one "$snapshot_url" "$snapshot"
}

pids=()
for realization in $(seq 0 26); do
    download_realization "$realization" &
    pids+=("$!")
    if [[ ${#pids[@]} -ge $workers ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

manifest=$root/download/hong2021_v14_astrid_raw_manifest.json
python src/hong2021_validate_astrid_raw.py \
    --seal "$seal" --repo "$repo" --root "$root" --out "$manifest"
printf '[complete] sealed Astrid CV0-26 raw manifest: %s\n' "$manifest"
