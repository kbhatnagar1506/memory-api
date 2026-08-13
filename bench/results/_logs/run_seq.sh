#!/bin/bash
# SEQUENTIAL, not parallel. Each arm holds the entire corpus in an
# InMemoryStore -- 373,687 chunks x 768 dims is ~2.3GB of raw vectors before
# Python's per-float overhead, and this machine has 8GB. Four at once was
# killed by memory pressure at [250/500]. The disk cache avoids re-embedding;
# it does nothing for RAM.
#
# Ordered by how much each answers, so the important number lands first.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
D=bench/results/_logs
FLAGS="--benchmark longmemeval --end-to-end --concurrency 24 --batch-size 32 --corpus-concurrency 6"

run () {
  local name="$1"; shift
  echo "=== $name started $(date +%H:%M:%S) ===" > "$D/$name.log"
  stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS "$@" --run-id "lme-v23-$name" 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> "$D/$name.log"
  echo "=== $name DONE $(date +%H:%M:%S) ===" >> "$D/$name.log"
}

run pro      --answer-model gemini-2.5-pro   # is the ceiling the model, or us?
run dynkonly --dynamic-k                     # was dynamic-k innocent?
run temporal                                 # the 0.910 retrieval cap
run think    --thinking-budget 2048          # reasoning effort, same model
echo "ALL DONE $(date +%H:%M:%S)" > "$D/seq.done"
