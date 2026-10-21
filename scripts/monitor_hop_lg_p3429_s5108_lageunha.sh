#!/usr/bin/env bash
set -u

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This monitor must execute on Lageunha." >&2
    exit 2
fi

work=/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1
hop_work="$work/hop_work"
status="$work/hop_gate_status.txt"
history="$work/hop_gate_monitor.log"
poll_seconds="${CF4_HOP_MONITOR_POLL_SECONDS:-60}"
iteration=0

mkdir -p "$work"
while true; do
    now="$(date --iso-8601=seconds)"
    state=unknown
    detail=unknown
    pid="$(pgrep -o -x hop || true)"
    elapsed=none
    cpu=0
    rss_kib=0
    if [[ -s "$work/GATE_COMPLETE" ]]; then
        state=complete
        detail=all_gates_complete
    elif [[ -s "$work/GATE_FAILED" || -s "$hop_work/HOP_FAILED" ]]; then
        state=failed
        detail=pipeline_failure_marker
    elif [[ -s "$hop_work/HOP_COMPLETE" ]]; then
        state=postprocessing
        detail=hop_complete_gate_running
    elif [[ -n "$pid" ]]; then
        state=running
        detail=hop_density_or_group_finding
        read -r elapsed cpu rss_kib < <(
            ps -o etime=,pcpu=,rss= -p "$pid" | awk '{print $1, $2, $3}')
    elif tmux has-session -t cf4_hop_5108_z0 2>/dev/null; then
        state=running
        detail=hop_regroup_stage
    else
        state=failed
        detail=no_hop_process_or_completion_marker
    fi
    tmp="${status}.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'detail=%s\n' "$detail"
        printf 'timestamp=%s\n' "$now"
        printf 'hop_pid=%s\n' "${pid:-none}"
        printf 'hop_elapsed=%s\n' "$elapsed"
        printf 'hop_cpu_percent=%s\n' "$cpu"
        printf 'hop_rss_kib=%s\n' "$rss_kib"
        for name in hop00010.den hop00010.hop hop00010.gbound \
                    grp00010.tag peaks00010.tag; do
            size="$(stat -c %s "$hop_work/$name" 2>/dev/null || printf 0)"
            printf '%s_bytes=%s\n' "${name//./_}" "$size"
        done
    } >"$tmp"
    mv -f "$tmp" "$status"
    if (( iteration % 10 == 0 )); then
        tr '\n' ' ' <"$status" >>"$history"
        printf '\n' >>"$history"
    fi
    if [[ "$state" == complete || "$state" == failed ]]; then
        exit 0
    fi
    iteration=$((iteration + 1))
    sleep "$poll_seconds"
done
