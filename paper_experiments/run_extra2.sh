#!/bin/bash
# Extra runs for the article, second version of the queue (2026-10-01). GPU1 only.
# Queue A (2 jobs in parallel): TCN / LSTM experiments of paper_extra.py.
# Queue B (2 jobs in parallel): Mamba autoencoder with the other sets of features.
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs scores
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
x() { n=$(echo "$*" | tr -d '-' | tr ' ' '_'); echo "python3 paper_extra.py $* > logs/x_${n}.log 2>&1"; }
{
  for s in 0 1 2; do x --model tcn --feat geom --common --seed $s; done
  for s in 0 1; do x --model tcn --feat hands --common --seed $s; x --model tcn --feat hh --common --seed $s; done
  x --model tcn --feat exh --seed 0
  x --model tcn --feat vit_geom --seed 0
  x --model lstm --feat geom --seq_len 12 --batch_size 128 --seed 0
  x --model tcn --feat exh --seed 1
  x --model tcn --feat vit_geom --seed 1
  x --model tcn --feat vit_coords --seed 0
  for s in 0 1; do for t in 30 60 90; do x --model tcn --feat geom --seq_len $t --min_clip_len 90 --seed $s; done; done
  for s in 0 1; do for z in 4 8 32; do x --model tcn --feat geom --latent $z --seed $s; done; done
  x --model lstm --feat exh --seq_len 12 --batch_size 128 --seed 0
  x --model lstm --feat geom --seq_len 12 --batch_size 128 --seed 1
  x --model tcn --feat vit_coords --seed 1
  x --model lstm --feat geom --seq_len 60 --batch_size 128 --seed 0
} > jobsA.txt
{
  for s in 0 1; do
    for spec in "coords:" "kin:" "geom:--common" "hands:--common" "hh:--common"; do
      f=${spec%%:*}; c=${spec#*:}; t=$f; [ -n "$c" ] && t=${f}_common
      for k in $(seq 0 13); do
        echo "python3 paper_mamba.py --feat $f $c --seed $s --only_fold $k --tag _f$k > logs/mamba_${t}_f${k}_s${s}.log 2>&1"
      done
    done
  done
} > jobsB.txt
xargs -P 2 -I{} bash -c "{}" < jobsA.txt &
xargs -P 2 -I{} bash -c "{}" < jobsB.txt &
wait
echo EXTRA_DONE > EXTRA_DONE.txt
