#!/bin/bash
# Held-out test (prereg_v2.md): waits for the replication to finish, then 49 tasks x base/elbi/elbi2.
cd "$(dirname "$0")"
while pgrep -f run_rep.sh > /dev/null; do sleep 60; done
for pass in 1 2 3 4; do
  RUNS=runs_heldout EXAM=exam_heldout.json ARMS="base elbi elbi2" WORKERS=4 python3 qr_run.py >> runs_heldout.log 2>&1
  n=$(ls runs_heldout/*/t*/answers/answer.txt 2>/dev/null | wc -l)
  echo "pass $pass: $n/147" >> runs_heldout.log
  [ "$n" -ge 147 ] && break
  sleep 600
done
