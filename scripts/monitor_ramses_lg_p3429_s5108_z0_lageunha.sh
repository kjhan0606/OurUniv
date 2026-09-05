#!/usr/bin/env bash
set -u

expected_host="lageunha"
actual_host="$(hostname | tr '[:upper:]' '[:lower:]')"
if [[ "$actual_host" != "$expected_host" ]]; then
    echo "This monitor must run on Lageunha, not $(hostname)." >&2
    exit 2
fi

run_dir=/gpfs/kjhan/CF4/ramses/lg_p3429_s5108_l12_l19_z0_v1
run_log="$run_dir/run.log"
monitor_log="$run_dir/completion_monitor.log"
status_file="$run_dir/completion_status.txt"
complete_marker="$run_dir/RAMSES_Z0_COMPLETE"
failed_marker="$run_dir/RAMSES_Z0_FAILED"
ready_marker="$run_dir/READY_FOR_HOP"
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax/ramses_lb_minimax3d
namelist=/home/kjhan/BACKUP/CF4/config/ramses_lg_p3429_s5108_pilot_z0_v1.nml
rank_pattern="^${binary} ${namelist}$"
poll_seconds="${CF4_MONITOR_POLL_SECONDS:-60}"

mkdir -p "$run_dir"
rm -f "$complete_marker" "$failed_marker" "$ready_marker"

write_running_status() {
    local now rank_count latest_line step aexp redshift output_count log_size tmp
    now="$(date --iso-8601=seconds)"
    rank_count="$(pgrep -fc "$rank_pattern" || true)"
    latest_line="$(grep 'Fine step=' "$run_log" 2>/dev/null | tail -1 || true)"
    step="$(sed -n 's/.*Fine step=[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<<"$latest_line")"
    aexp="$(sed -n 's/.*a=[[:space:]]*\([0-9.Ee+-]*\).*/\1/p' <<<"$latest_line")"
    if [[ -n "$aexp" ]]; then
        redshift="$(awk -v a="$aexp" 'BEGIN {printf "%.8g", 1.0/a-1.0}')"
    else
        redshift="unknown"
    fi
    output_count="$(find "$run_dir" -maxdepth 1 -type d -name 'output_*' 2>/dev/null | wc -l)"
    log_size="$(stat -c %s "$run_log" 2>/dev/null || printf '0')"
    tmp="${status_file}.tmp.$$"
    {
        printf 'state=running\n'
        printf 'timestamp=%s\n' "$now"
        printf 'rank_count=%s\n' "$rank_count"
        printf 'fine_step=%s\n' "${step:-unknown}"
        printf 'aexp=%s\n' "${aexp:-unknown}"
        printf 'redshift=%s\n' "$redshift"
        printf 'output_count=%s\n' "$output_count"
        printf 'run_log_bytes=%s\n' "$log_size"
    } >"$tmp"
    mv -f "$tmp" "$status_file"
}

write_final_status() {
    local state reason output aexp part_count amr_count fatal_count now tmp
    state="$1"
    reason="$2"
    output="$3"
    aexp="$4"
    part_count="$5"
    amr_count="$6"
    fatal_count="$7"
    now="$(date --iso-8601=seconds)"
    tmp="${status_file}.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'timestamp=%s\n' "$now"
        printf 'reason=%s\n' "$reason"
        printf 'final_output=%s\n' "$output"
        printf 'aexp=%s\n' "$aexp"
        printf 'particle_file_count=%s\n' "$part_count"
        printf 'amr_file_count=%s\n' "$amr_count"
        printf 'fatal_pattern_count=%s\n' "$fatal_count"
    } >"$tmp"
    mv -f "$tmp" "$status_file"
}

printf '%s monitor attached; poll=%ss\n' \
    "$(date --iso-8601=seconds)" "$poll_seconds" >>"$monitor_log"

poll_number=0
while pgrep -f "$rank_pattern" >/dev/null; do
    write_running_status
    if (( poll_number % 10 == 0 )); then
        tr '\n' ' ' <"$status_file" >>"$monitor_log"
        printf '\n' >>"$monitor_log"
    fi
    poll_number=$((poll_number + 1))
    sleep "$poll_seconds"
done

# Let the parallel filesystem settle after all MPI ranks disappear.
sleep 10

latest_output="$(find "$run_dir" -maxdepth 1 -type d -name 'output_*' \
    -printf '%p\n' 2>/dev/null | sort -V | tail -1)"
aexp="unknown"
part_count=0
amr_count=0
if [[ -n "$latest_output" ]]; then
    output_number="${latest_output##*_}"
    info_file="$latest_output/info_${output_number}.txt"
    if [[ -r "$info_file" ]]; then
        aexp="$(awk '$1 == "aexp" {print $3; exit}' "$info_file")"
    fi
    part_count="$(find "$latest_output" -maxdepth 1 -type f \
        -name "part_${output_number}.out*" | wc -l)"
    amr_count="$(find "$latest_output" -maxdepth 1 -type f \
        -name "amr_${output_number}.out*" | wc -l)"
fi
fatal_count="$(grep -Eic 'fatal|segmentation|out of memory|MPI_ABORT' \
    "$run_log" 2>/dev/null || true)"

final_aexp_ok=0
if [[ "$aexp" != "unknown" ]]; then
    final_aexp_ok="$(awk -v a="$aexp" 'BEGIN {print (a >= 0.999999) ? 1 : 0}')"
fi

if [[ "$final_aexp_ok" == 1 && "$part_count" == 16 && "$amr_count" == 16 \
      && "$fatal_count" == 0 ]]; then
    write_final_status complete z0_snapshot_valid "$latest_output" "$aexp" \
        "$part_count" "$amr_count" "$fatal_count"
    printf '%s RAMSES z=0 snapshot validated: %s\n' \
        "$(date --iso-8601=seconds)" "$latest_output" >>"$monitor_log"
    cp -f "$status_file" "$complete_marker"
    cp -f "$status_file" "$ready_marker"
    exit 0
fi

reason="ranks_exited_without_complete_z0_snapshot"
write_final_status failed "$reason" "${latest_output:-none}" "$aexp" \
    "$part_count" "$amr_count" "$fatal_count"
printf '%s RAMSES completion validation FAILED: output=%s aexp=%s part=%s amr=%s fatal=%s\n' \
    "$(date --iso-8601=seconds)" "${latest_output:-none}" "$aexp" \
    "$part_count" "$amr_count" "$fatal_count" >>"$monitor_log"
cp -f "$status_file" "$failed_marker"
exit 1
