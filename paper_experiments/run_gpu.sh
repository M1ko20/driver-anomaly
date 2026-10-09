#!/bin/bash
# GPU queue for the ablation runs (2026-10-09). GPUs 1-3 (never GPU0), <slots> jobs per GPU;
# every job is limited to a small fraction of the GPU memory (--gpu_mem_frac in the job
# line), because the GPUs are shared with the jobs of other users. A job is one fold of one
# run; jobs whose log already exists are skipped (the same convention as run_capped.sh, so
# both queues can work on the same list of runs).
#   ./run_gpu.sh <jobs file> <slots per GPU>
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=$1
worker() {
  while true; do
    job=$( (
      flock 9
      while read -r l; do
        [ -z "$l" ] && continue
        log=$(echo "$l" | sed -E 's/.*> (logs\/[^ ]+).*/\1/')
        if [ ! -e "$log" ]; then touch "$log"; echo "$l"; break; fi
      done < "$JOBS"
    ) 9> gpu_queue.lock )
    [ -z "$job" ] && break
    CUDA_VISIBLE_DEVICES=$1 bash -c "$job" < /dev/null
  done
}
for g in 1 2 3; do for s in $(seq "$2"); do worker $g & sleep 2; done; done
wait
echo "DONE $(date)" > "$(basename "$1" .txt)_DONE.txt"
