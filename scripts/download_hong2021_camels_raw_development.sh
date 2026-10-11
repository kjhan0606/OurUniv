#!/usr/bin/env bash
set -euo pipefail

# Download only V14 development suites allowed by the committed v2 firewall.
# Astrid is deliberately rejected here and has no download path until the
# complete V14 model and one-shot independent command are frozen.

suite=${1:?usage: download_hong2021_camels_raw_development.sh SIMBA|Swift-EAGLE [root]}
case "$suite" in
    SIMBA)
        first=16
        last=26
        ;;
    Swift-EAGLE)
        first=0
        last=26
        ;;
    *)
        printf 'Refusing non-development suite: %s\n' "$suite" >&2
        exit 2
        ;;
esac

root=${2:-/gpfs/kjhan/CAMELS/$suite/L25n256}
public=https://users.flatironinstitute.org/~camels
workers=${CAMELS_DOWNLOAD_WORKERS:-3}
if [[ ! $workers =~ ^[1-9][0-9]*$ ]]; then
    printf 'CAMELS_DOWNLOAD_WORKERS must be a positive integer\n' >&2
    exit 2
fi
mkdir -p "$root/raw" "$root/CV" "$root/download"

remote_bytes() {
    local url=$1
    local value
    value=$(curl --fail --location --silent --show-error --head "$url" \
        | awk 'BEGIN{IGNORECASE=1} /^content-length:/{gsub("\r",""); n=$2} END{print n}')
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        printf 'Cannot determine Content-Length for %s\n' "$url" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

download_one() {
    local url=$1
    local destination=$2
    local expected current partial
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
    curl --fail --location --retry 20 --retry-delay 5 \
        --continue-at - --silent --show-error --output "$partial" "$url"
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
    local snapshot_url snapshot catalog_url catalog
    snapshot_url=$public/Sims/$suite/L25n256/CV/CV_${realization}/snapshot_090.hdf5
    catalog_url=$public/FOF_Subfind/$suite/L25n256/CV/CV_${realization}/groups_090.hdf5
    snapshot=$root/raw/CV_${realization}/snapshot_090.hdf5
    catalog=$root/CV/CV_${realization}/groups_090.hdf5
    download_one "$catalog_url" "$catalog"
    download_one "$snapshot_url" "$snapshot"
}

pids=()
for realization in $(seq "$first" "$last"); do
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

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHONPATH=$repo/src python "$repo/src/hong2021_validate_camels_raw.py" \
    --suite "$suite" --root "$root" --first "$first" --last "$last" \
    --out "$root/download/hong2021_v14_raw_development_manifest.json"
