#!/bin/bash
# Extra runs for the article (started 2026-10-01): second runs of the variants that
# were left with one run, and the Mamba autoencoder with the other sets of features.
# GPU1 only (GPU0 is shared, GPU2/3 are used by another user), 4 jobs in parallel.
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs scores
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
{
  for e in coords_tcn forecast_tcn stgcn_coords stgcn_kin; do
    echo "python3 paper_loso.py --exp $e --seed 1 > logs/${e}_s1.log 2>&1"
  done
  for s in 0 1; do
    for spec in "coords:" "kin:" "geom:--common" "hands:--common" "hh:--common"; do
      f=${spec%%:*}; c=${spec#*:}; t=$f; [ -n "$c" ] && t=${f}_common
      for k in $(seq 0 13); do
        echo "python3 paper_mamba.py --feat $f $c --seed $s --only_fold $k --tag _f$k > logs/mamba_${t}_f${k}_s${s}.log 2>&1"
      done
    done
  done
} > extra_jobs.txt
xargs -P 4 -I{} bash -c "{}" < extra_jobs.txt
echo EXTRA_DONE > EXTRA_DONE.txt
