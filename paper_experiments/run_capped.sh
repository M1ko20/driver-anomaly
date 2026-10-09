#!/bin/bash
# Runs the jobs of a job file (one fold of one run per line) on CPU with low priority.
# At most `cat cap.txt` paper_ablation.py processes run at once (the cap can be changed
# while the queue is running). Jobs whose log already exists are skipped.
#   ./run_capped.sh <jobs file>
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while read -r l; do
  [ -z "$l" ] && continue
  log=$(echo "$l" | sed -E 's/.*> (logs\/[^ ]+).*/\1/')
  [ -e "$log" ] && continue
  while [ "$(pgrep -u "$USER" -fc '^python3 paper_ablation.py')" -ge "$(cat cap.txt)" ]; do sleep 30; done
  nice -n 10 bash -c "$l" < /dev/null &
  sleep 3
done < "$1"
wait
echo "DONE $(date)" > "$(basename "$1" .txt)_DONE.txt"
