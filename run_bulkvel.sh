#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/bulkvel.log
echo "=== LG bulk vel gate $(date +%H:%M) ===" > $LOG
$PY -u src/cf4_lg_bulkvel.py --recon recon/cf4_map_cr6.npz --seed 19 --label cr6_e19 >> $LOG 2>&1
echo "BULK_DONE $(date +%H:%M)" >> $LOG
