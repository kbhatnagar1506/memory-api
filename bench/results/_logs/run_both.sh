#!/bin/bash
# Both arms, back to back, in a script on disk so no shell teardown can orphan
# the second one -- which is exactly what happened the first time these were
# chained from an inline `nohup bash -c`.
#
# --concurrency 32 --batch-size 200: embedding is network-bound, and the token
# batcher already caps requests at 15,000 tokens, so a big item count only
# affects the SHORT texts. Claims are ~15 tokens each, where a 32-item cap was
# paying one round trip per 480 tokens against a 15,000 budget.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/extract-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 200"

echo "=== ADD started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract add --run-id lme-v17-extract-add 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG

echo "=== ONLY started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract only --run-id lme-v18-extract-only 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG

echo "=== DONE $(date +%H:%M:%S) ===" >> $LOG
