#!/bin/bash
# Ablation queue for the article (2026-10-09). CPU only (the GPUs are used by other
# users), niced, one thread per job; a job is one fold of one run. Jobs whose log
# already exists are skipped, so the queue can be restarted.
#   ./run_ablation.sh <jobs file> <parallel jobs> <threads per job>
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$3 MKL_NUM_THREADS=$3 OPENBLAS_NUM_THREADS=$3
grep -v '^$' "$1" | while read -r l; do
  log=$(echo "$l" | sed -E 's/.*> (logs\/[^ ]+).*/\1/')
  [ -e "$log" ] || echo "$l"
done | xargs -P "$2" -d '\n' -I{} nice -n 10 bash -c "{}"
echo "DONE $(date)" > "$(basename "$1" .txt)_DONE.txt"
