#!/bin/bash
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
run() { g=$1; exp=$2; seed=$3; CUDA_VISIBLE_DEVICES=$g python3 paper_loso.py --exp $exp --seed $seed >> logs/${exp}_s${seed}.log 2>&1; }
run 3 coords_tcn 0 &
run 2 forecast_tcn 0 &
run 1 stgcn_kin 0 &
wait
echo REST_DONE > REST_DONE.txt
