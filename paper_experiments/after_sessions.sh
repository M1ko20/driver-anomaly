#!/bin/bash
# Chain that starts when the session-level pose extraction (pipeline/run_pose_sessions.sh) is
# finished: (1) clip features with the corrected alignment (beta_raw_v3) and the control
# (beta_raw_v2s), (2) OC-SVM on the averaged features with both, (3) scores on the
# uninterrupted recordings, (4) the autoencoders with the corrected alignment.
# sessions_v2/CHAIN_DONE.txt is written after step (3).
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542
while [ ! -e sessions_v2/DONE.txt ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 pipeline/build_beta_from_sessions.py > sessions_v2/logs/build.log 2>&1 || { echo "FAIL build $(date)" > sessions_v2/CHAIN_DONE.txt; exit 1; }
cd paper_experiments
: > jobs_v3_classical.txt
for raw in beta_raw_v3 beta_raw_v2s; do for f in geom_da exh geom; do for s in 0 1 2; do
  echo "python3 paper_classical.py --feat $f --pool mean --seed $s --raw $raw --methods ocsvm > logs/c_${raw}_${f}_s$s.log 2>&1" >> jobs_v3_classical.txt
done; done; done
./run_ablation.sh jobs_v3_classical.txt 9 1
: > jobs_cont.txt
for f in geom_da exh; do
  m=ocsvm,ocsvm_frame; [ "$f" = exh ] && m=ocsvm,tcn
  for k in $(seq 0 13); do
    echo "python3 paper_continuous.py --feat $f --methods $m --only_fold $k --threads 2 > logs/cont_${f}_f$(printf %02d $k).log 2>&1" >> jobs_cont.txt
  done
done
./run_ablation.sh jobs_cont.txt 14 2
echo "OK $(date)" > ../sessions_v2/CHAIN_DONE.txt
: > jobs_v3_tcn.txt
for f in geom_da exh; do for k in $(seq 0 13); do
  echo "python3 paper_ablation.py --feat $f --kernel 1 --pool mean --seed 0 --only_fold $k --threads 1 --raw beta_raw_v3 > logs/a_v3_mean_${f}_s0_f$(printf %02d $k).log 2>&1" >> jobs_v3_tcn.txt
done; done
for k in $(seq 0 13); do
  echo "python3 paper_ablation.py --feat geom_da --seed 0 --only_fold $k --threads 1 --raw beta_raw_v3 > logs/a_v3_geom_da_s0_f$(printf %02d $k).log 2>&1" >> jobs_v3_tcn.txt
done
./run_capped.sh jobs_v3_tcn.txt
