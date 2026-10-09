#!/bin/bash
# Extra runs for the article, third version of the queue (2026-10-01): GPU1-3,
# 10 jobs in parallel in total, jobs that were already started are skipped.
source /home/fei/mik0542/yolo_env/bin/activate
cd /home/fei/mik0542/paper_experiments
python3 - <<'PY'
import os, re
a = [l.strip() for l in open("jobsA.txt") if l.strip()]
b = [l.strip() for l in open("jobsB.txt") if l.strip()]
started = lambda l: os.path.exists(re.search(r"> (logs/\S+)", l).group(1))
a = [l for l in a if not started(l)]
b = [l for l in b if not started(l)]
out = []
while a or b:
    if a: out.append(a.pop(0))
    out.extend(b[:2]); del b[:2]
open("jobs3.txt", "w").write("\n".join(out) + "\n")
print(len(out), "jobs")
PY
xargs -P 10 -d '\n' -I{} ./run_job.sh "{}" < jobs3.txt
echo EXTRA_DONE > EXTRA_DONE.txt
