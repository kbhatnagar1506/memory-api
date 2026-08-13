#!/bin/bash
# Two arms back to back, in a script on disk so no shell teardown orphans the
# second one.
#
# BASELINE first and it is nearly free: every vector is already in the disk
# cache, so it costs the answer path only. CONTEXTUAL then pays for 500
# corpora of fresh embeddings, because the cache is keyed on the embed INPUT
# and the header changes it -- which is the point, and is what makes the two
# numbers attributable to one variable.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/ctx-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 32 --corpus-concurrency 8"

echo "=== BASELINE started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --run-id lme-v21-baseline 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> $LOG

echo "=== CONTEXTUAL started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --contextual-embedding --run-id lme-v21-contextual 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> $LOG

echo "=== DONE $(date +%H:%M:%S) ===" >> $LOG
