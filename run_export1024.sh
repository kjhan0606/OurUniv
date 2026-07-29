#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/export1024.log
echo "=== GRAFIC export cr6 e19 @ 1024^3 (CPU, level_010) $(date +%H:%M) ===" > $LOG
$PY -u src/cf4_export_grafic.py --s-npy recon/s_cr6_e19_1024.npy --N 1024 --spacing 0.375 \
    --Om 0.31 --h 0.746 --A-s-1e9 1.63 --astart 0.02 \
    --out recon/ic_cr6_e19_1024/level_010 >> $LOG 2>&1
echo "EXPORT1024_DONE $(date +%H:%M)" >> $LOG
