#!/usr/bin/env bash
# Download and verify the EAGLE RefL0100N1504 z=0 particle snapshot.

set -euo pipefail

root=${EAGLE_ROOT:-/gpfs/kjhan/EAGLE/RefL0100N1504}
netrc=${VIRGO_NETRC:-$HOME/.config/virgodb/netrc}
url=https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots/download?run=RefL0100N1504\&snapnum=28
expected_bytes=504327096320
raw_dir=$root/raw
archive=$raw_dir/RefL0100N1504_snap_028.tar
partial=$archive.partial
manifest=$raw_dir/RefL0100N1504_snap_028.tar.list

if [[ ! -r "$netrc" ]]; then
    printf 'Missing VirgoDB credential file: %s\n' "$netrc" >&2
    exit 2
fi

mkdir -p "$raw_dir"
exec 9>"$root/.download_ref100_z0.lock"
if ! flock -n 9; then
    printf 'Another RefL0100N1504 z=0 download is already running.\n' >&2
    exit 3
fi

if [[ -f "$archive" ]]; then
    actual_bytes=$(stat -c %s "$archive")
    if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
        printf 'Existing archive has %s bytes; expected %s.\n' \
            "$actual_bytes" "$expected_bytes" >&2
        exit 4
    fi
else
    if [[ -f "$partial" ]] && [[ $(stat -c %s "$partial") -gt "$expected_bytes" ]]; then
        printf 'Partial archive is larger than the expected payload.\n' >&2
        exit 5
    fi

    attempt=0
    while :; do
        current_bytes=0
        [[ -f "$partial" ]] && current_bytes=$(stat -c %s "$partial")
        if [[ "$current_bytes" -eq "$expected_bytes" ]]; then
            break
        fi
        attempt=$((attempt + 1))
        printf '[download] attempt=%d bytes=%d/%d timestamp=%s\n' \
            "$attempt" "$current_bytes" "$expected_bytes" \
            "$(date --iso-8601=seconds)"
        if ! curl --fail --show-error --silent --location \
            --netrc-file "$netrc" --continue-at - --output "$partial" \
            --connect-timeout 30 --speed-limit 1024 --speed-time 300 \
            "$url"; then
            printf '[download] curl failed; retrying in 30 seconds\n' >&2
            sleep 30
            continue
        fi
        current_bytes=$(stat -c %s "$partial")
        if [[ "$current_bytes" -gt "$expected_bytes" ]]; then
            printf 'Downloaded payload is larger than expected: %s bytes\n' \
                "$current_bytes" >&2
            exit 6
        fi
        [[ "$current_bytes" -eq "$expected_bytes" ]] && break
        printf '[download] short payload (%d bytes); retrying in 30 seconds\n' \
            "$current_bytes" >&2
        sleep 30
    done
    mv -f "$partial" "$archive"
fi

actual_bytes=$(stat -c %s "$archive")
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    printf 'Final size mismatch: %s != %s\n' "$actual_bytes" "$expected_bytes" >&2
    exit 7
fi

printf '[verify] archive size is exact; scanning tar structure\n'
tar -tf "$archive" >"$manifest.partial"
mv -f "$manifest.partial" "$manifest"
member_count=$(wc -l <"$manifest")
if [[ "$member_count" -lt 1 ]]; then
    printf 'Archive contains no members.\n' >&2
    exit 8
fi

printf '%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$actual_bytes" "$member_count" "$archive" \
    >"$root/download_ref100_z0.complete"
printf '[complete] bytes=%s tar_members=%s archive=%s\n' \
    "$actual_bytes" "$member_count" "$archive"
