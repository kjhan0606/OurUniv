#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/export_cr6b.log
echo "=== RE-export cr6 e19 with CORRECT CF4 cosmology $(date +%H:%M) ===" > $LOG
$PY -u src/cf4_export_grafic.py --s-npy recon/s_cr6_e19_576.npy --N 576 --spacing 0.666667 \
    --Om 0.31 --h 0.746 --A-s-1e9 1.63 --astart 0.02 --out recon/ic_cr6_e19 >> $LOG 2>&1
echo "EXPORT_DONE $(date +%H:%M)" >> $LOG
