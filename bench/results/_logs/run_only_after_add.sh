#!/bin/bash
# Waits for the `add` arm to exit, then runs `only`. Separate script on disk so
# it does not die with the shell that launched it -- the first attempt at
# chaining these lost its wrapper to session teardown and the second run would
# never have started.
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
while pgrep -f "extract add --run-id lme-v17" >/dev/null; do sleep 30; done
echo "=== ADD finished, ONLY started $(date +%H:%M:%S) ===" >> bench/results/_logs/extract-full.log
stdbuf -oL -eL .venv/bin/python -u -m bench.run --benchmark longmemeval --end-to-end \
    --extract only --run-id lme-v18-extract-only 2>&1 \
    | grep --line-buffered -vE "^AFC" >> bench/results/_logs/extract-full.log
echo "=== DONE $(date +%H:%M:%S) ===" >> bench/results/_logs/extract-full.log
