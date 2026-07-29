#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
rm -f logs/lg_stage2_done.flag
# top completion seeds from the linear screen x a few fine seeds
for cs in 108 92 93 73 128; do
  for fs in 1 2; do
    echo "=== cseed=$cs fseed=$fs ==="
    $PY -u src/cf4_lg_search.py --cseed "$cs" --fseed "$fs" --Nfine 576 \
      --outdir recon/lg_search > "logs/lg_c${cs}f${fs}_576.log" 2>&1
  done
done
echo ALL_DONE > logs/lg_stage2_done.flag
