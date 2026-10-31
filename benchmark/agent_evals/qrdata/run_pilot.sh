#!/bin/bash
# Up to 4 passes; each skips tasks that already have an answer (survives usage-limit stops).
cd "$(dirname "$0")"
for pass in 1 2 3 4; do
  RUNS=runs_pilot ARMS="base elbi" WORKERS=4 python3 qr_run.py >> runs_pilot.log 2>&1
  n=$(ls runs_pilot/*/t*/answers/answer.txt 2>/dev/null | wc -l)
  echo "pass $pass done: $n/80 answered" >> runs_pilot.log
  [ "$n" -ge 80 ] && break
  sleep 600
done
