#!/bin/bash
# Phase-B re-screen only (cr2-8 generation already done). pair_score bug fixed.
# cr{1-8} x embed{1-24}, one process per (cr,embed) to avoid the 32 GB Ada OOM. Fresh TSV.
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/rescreen.log
TSV=recon/screen3c_rows.tsv
echo "=== RESCREEN cr{1-8} x embed{1-24} $(date +%H:%M) ===" > $LOG
rm -f $TSV
for cs in 1 2 3 4 5 6 7 8; do
  if [ $cs -eq 1 ]; then RC=recon/cf4_map_cf4_real192_hr.npz; else RC=recon/cf4_map_cr${cs}.npz; fi
  [ -f "$RC" ] || { echo "MISSING $RC, skip cr$cs" >> $LOG; continue; }
  for es in $(seq 1 24); do
    $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed $es --tsv $TSV >> $LOG 2>&1
  done
done
echo "SCREEN_DONE $(date +%H:%M)" >> $LOG
$PY -u src/cf4_lg_screen3.py --rank-only --tsv $TSV >> $LOG 2>&1
echo "RESCREEN_DONE $(date +%H:%M)" >> $LOG
