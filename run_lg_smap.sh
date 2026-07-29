#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
rm -f logs/lg_smap_done.flag
# fixed correct environment (s_map, Virgo preserved); scan fine seeds for the LG pair
for fs in $(seq 2 20); do
  echo "=== s_map fseed=$fs ==="
  $PY -u src/cf4_lg_search.py --field s_map --fseed "$fs" --Nfine 576 \
    --outdir recon/lg_search > "logs/lg_smapf${fs}.log" 2>&1
done
echo ALL_DONE > logs/lg_smap_done.flag
