#!/bin/bash
# TWO at a time, not four. Each arm holds the whole corpus in an
# InMemoryStore and four of them were killed by memory pressure at [250/500]
# on this 8GB machine. Two is the most that has been shown to fit.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
D=bench/results/_logs
FLAGS="--benchmark longmemeval --end-to-end --concurrency 16 --batch-size 32 --corpus-concurrency 4"

run () {
  local name="$1"; shift
  echo "=== $name started $(date +%H:%M:%S) ===" > "$D/$name.log"
  stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS "$@" --run-id "lme-v24-$name" 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> "$D/$name.log"
  echo "=== $name DONE $(date +%H:%M:%S) ===" >> "$D/$name.log"
}

# Pair 1: the new reference, and the arm whose verdict the truncation fix changes.
run base2   &
run derive2 --derive --dynamic-k &
wait
# Pair 2: the arm whose verdict the token fix changes, plus the untested lever.
run pro2      --answer-model gemini-2.5-pro --thinking-budget 1024 &
run temporal2 &
wait
echo "ALL DONE $(date +%H:%M:%S)" > "$D/fixed.done"
