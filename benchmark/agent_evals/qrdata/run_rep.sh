#!/bin/bash
# Replication: pilot tasks 11-20 (Causality/numerical), 3 more rounds, base vs elbi (v1 tool, unchanged).
cd "$(dirname "$0")"
for pass in 1 2 3; do
  for r in 1 2 3; do
    RUNS=runs_rep$r ARMS="base elbi" WORKERS=4 python3 qr_run.py 11 12 13 14 15 16 17 18 19 20 >> runs_rep.log 2>&1
  done
  n=$(ls runs_rep*/*/t*/answers/answer.txt 2>/dev/null | wc -l)
  echo "pass $pass: $n/60" >> runs_rep.log
  [ "$n" -ge 60 ] && break
  sleep 600
done
