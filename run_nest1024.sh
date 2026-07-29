#!/bin/bash
cd /home/kjhan/BACKUP/CF4
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/nest1024.log
echo "=== nest 576->1024 cr6 e19 $(date +%H:%M) ===" > $LOG
$PY -u src/cf4_nest_1024.py --f576 recon/s_cr6_e19_576.npy --Nf 1024 --seed2 2019 \
    --out recon/s_cr6_e19_1024.npy >> $LOG 2>&1
echo "NEST_DONE $(date +%H:%M)" >> $LOG
