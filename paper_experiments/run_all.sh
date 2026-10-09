#!/bin/bash
# Runs all paper experiments (GPU1-3 only; GPU0 is shared with other users).
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs
run() { g=$1; exp=$2; seed=$3; CUDA_VISIBLE_DEVICES=$g python3 paper_loso.py --exp $exp --seed $seed > logs/${exp}_s${seed}.log 2>&1; }
chain1() { run 1 geom_tcn 0; run 1 geom_tcn 1; run 1 coords_tcn 0; run 1 coords_tcn 1; }
chain2() { run 2 kin_tcn 0; run 2 kin_tcn 1; run 2 forecast_tcn 0; run 2 forecast_tcn 1; }
chain3() { run 3 stgcn_coords 0; run 3 stgcn_kin 0; run 3 stgcn_coords 1; run 3 stgcn_kin 1; }
mamba() {
  for s in 0 1; do
    seq 0 13 | xargs -P 6 -I{} bash -c 'g=$(( ({} % 3) + 1 )); CUDA_VISIBLE_DEVICES=$g python3 paper_loso.py --exp mambalike_lt --seed '$s' --only_fold {} --tag _f{} > logs/mambalike_lt_f{}_s'$s'.log 2>&1'
  done
}
base() { for s in 0 1 2; do OMP_NUM_THREADS=6 python3 paper_baselines.py --feat $1 --seed $s > logs/baselines_$1_s$s.log 2>&1; done; }
chain1 & chain2 & chain3 & mamba & base coords & base geom & base kin &
wait
echo ALL_DONE > ALL_DONE.txt
