#!/bin/bash
# Full official CausalDS realistic exam (100 tasks), arms base and elbi, rounds 1-3.
# Resumes after usage limits: the runner skips tasks that already have an answer.
cd "$(dirname "$0")"
PY=${CAUSALDS_DIR:-upstream}/.venv/bin/python
missing() {
  n=0
  for r in 1 2 3; do for a in base elbi; do
    c=$(ls runs_official_r$r/$a/t*/answers/* 2>/dev/null | wc -l)
    n=$((n + 100 - c))
  done; done
  echo $n
}
for attempt in $(seq 1 40); do
  for r in 1 2 3; do
    RUNS=runs_official_r$r EXAM=exam_official.json ARMS="base elbi" WORKERS=4 $PY cds_run.py >> run-official.log 2>&1
  done
  m=$(missing)
  echo "$(date) attempt $attempt: $m answers missing" >> run-official.log
  [ "$m" -le 6 ] && [ "$attempt" -ge 2 ] && break
  if grep -l usage_limit_reached runs_official_r*/*/t*/transcript.jsonl >/dev/null 2>&1; then
    sleep 1800
  fi
done
echo "$(date) done: $(missing) missing" >> run-official.log
