#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]]; then
    echo "This watcher must run on Lageunha." >&2
    exit 2
fi

while pgrep -f '/zoom_run_sidm[3]/ramses_zoom3d' >/dev/null; do
    echo "$(date --iso-8601=seconds) waiting: SIDM3 ranks still own cores 0-31"
    sleep 30
done

echo "$(date --iso-8601=seconds) SIDM3 exited; starting the LG z=0 pilot"
exec bash /home/kjhan/BACKUP/CF4/scripts/run_ramses_lg_p3429_s5108_z0_lageunha.sh
