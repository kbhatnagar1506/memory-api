#!/bin/bash
# The final arm: extraction + per-source cap + kind routing. If the diagnosis
# is right this is the one that beats baseline -- claims keep the semantic
# gain, the cap restores coverage, and routing gives episodic questions their
# narrative back.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
LOG=bench/results/_logs/extract-full.log
FLAGS="--benchmark longmemeval --end-to-end --concurrency 32 --batch-size 32 --corpus-concurrency 8"

while pgrep -f "max-per-source 2 --run-id lme-v19" >/dev/null; do sleep 20; done

echo "=== ROUTING arm started $(date +%H:%M:%S) ===" >> $LOG
stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS \
    --extract add --max-per-source 2 --route-by-kind \
    --run-id lme-v20-extract-cap-routed 2>&1 \
    | grep --line-buffered -vE "^AFC" >> $LOG
echo "=== ALL ARMS DONE $(date +%H:%M:%S) ===" >> $LOG
