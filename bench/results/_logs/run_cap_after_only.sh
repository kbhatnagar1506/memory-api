#!/bin/bash
# Waits for the `only` arm, then runs the per-source cap arm on `add` storage.
# On disk, not an inline nohup: an earlier inline chain lost its wrapper to
# session teardown and the queued run silently never started.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/extract-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 32 --corpus-concurrency 8"

while pgrep -f "extract only --run-id lme-v18" >/dev/null; do sleep 20; done

echo "=== CAP arm started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract add --max-per-source 2 --run-id lme-v19-extract-add-cap2 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG
echo "=== ALL DONE $(date +%H:%M:%S) ===" >> $LOG
