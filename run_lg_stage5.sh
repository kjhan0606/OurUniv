#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
rm -f logs/lg_stage5_done.flag
# env-screen pool (good Virgo geometry + Local Void) x fine seeds; infall/approach decided by fine seed
for cs in 91 7 146 97 8 174 166 41; do
  for fs in 1 2 3; do
    echo "=== cseed=$cs fseed=$fs ==="
    $PY -u src/cf4_lg_search.py --cseed "$cs" --fseed "$fs" --Nfine 576 \
      --outdir recon/lg_search > "logs/lg_c${cs}f${fs}_576.log" 2>&1
  done
done
echo ALL_DONE > logs/lg_stage5_done.flag
