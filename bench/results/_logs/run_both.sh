#!/bin/bash
# Both arms back to back, in a script on disk so no shell teardown orphans the
# second one.
#
# Flag shape matters: batch_size 32 gives ~52 embedding requests per corpus,
# which is enough to fill --concurrency; batch_size 200 gave 8 and left 75% of
# the slots idle. --corpus-concurrency then keeps them filled across corpus
# boundaries instead of draining at each one.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/extract-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 32 --corpus-concurrency 8"

echo "=== ADD started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract add --run-id lme-v17-extract-add 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG

echo "=== ONLY started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract only --run-id lme-v18-extract-only 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG

echo "=== DONE $(date +%H:%M:%S) ===" >> $LOG
