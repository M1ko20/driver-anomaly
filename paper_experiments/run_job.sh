#!/bin/bash
# Runs one job (command line in $1) on the GPU (1, 2 or 3; never GPU0) that currently
# has the fewest of our jobs. The assignment is tracked by files slots/<gpu>/<pid>.
cd /home/fei/mik0542/paper_experiments
S=slots; mkdir -p $S/1 $S/2 $S/3
exec 9>$S/lock; flock 9
best=; bn=999
for g in 2 3 1; do
  for f in $S/$g/*; do [ -e "$f" ] && ! kill -0 "$(basename "$f")" 2>/dev/null && rm -f "$f"; done
  n=$(ls $S/$g | wc -l)
  if [ "$n" -lt "$bn" ]; then bn=$n; best=$g; fi
done
touch $S/$best/$$; flock -u 9
CUDA_VISIBLE_DEVICES=$best OMP_NUM_THREADS=4 bash -c "$1"
rm -f $S/$best/$$
