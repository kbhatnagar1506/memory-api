#!/bin/bash
# The lever aimed at where the losses actually are.
#
# 62 of ~86 lost questions sit in multi-session (73.5%, retrieval 0.970) and
# temporal-reasoning (79.7%, retrieval 0.910) -- the evidence is retrieved and
# the model fails to count, order and combine it. --derive routes those shapes
# through map->ground->reduce so code does the arithmetic and the model only
# locates instances. --dynamic-k sizes the evidence budget by question shape
# instead of a flat k=10; multi-session averages 2.59 evidence sessions.
#
# Cheap now: every embedding is cached, so this costs the answer path only.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/derive-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 32 --corpus-concurrency 8"

echo "=== DERIVE+DYNAMIC-K started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --derive --dynamic-k --run-id lme-v22-derive 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> $LOG

echo "=== DONE $(date +%H:%M:%S) ===" >> $LOG
