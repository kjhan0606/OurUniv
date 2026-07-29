#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
rm -f logs/lg576_done.flag
# combos: cseed fseed  (c1f1 already run separately)
for combo in "1 2" "8 1" "8 2" "19 1" "19 2"; do
  set -- $combo; cs=$1; fs=$2
  echo "=== running cseed=$cs fseed=$fs ==="
  $PY -u src/cf4_lg_search.py --cseed "$cs" --fseed "$fs" --Nfine 576 \
    --outdir recon/lg_search > "logs/lg_c${cs}f${fs}_576.log" 2>&1
done
echo ALL_DONE > logs/lg576_done.flag
