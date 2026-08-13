#!/bin/bash
# Four arms at once. Every embedding is already cached, so each arm's cost is
# the ANSWER path -- which is network-bound on Vertex, not CPU-bound here.
# Serially this is ~100 minutes; in parallel it is bounded by the slowest arm.
#
# Concurrency dropped 32 -> 12 per arm: four arms x 32 would be 128 in-flight
# requests against one Vertex quota, and rate-limit retries would make the
# whole thing slower than running them one at a time.
#
# ONE VARIABLE PER ARM against the 0.827 baseline:
#   dynkonly  --dynamic-k          (never measured alone; was bundled with derive)
#   pro       --answer-model pro   (is the ceiling the model or us?)
#   think     --thinking-budget    (reasoning effort, same model)
#   temporal  (code change already committed; asked_at now reaches retrieval)
export GOOGLE_CLOUD_PROJECT=patchguard-reakon
cd /Users/krishnabhatnagar/mapi
D=bench/results/_logs
FLAGS="--benchmark longmemeval --end-to-end --concurrency 12 --batch-size 32 --corpus-concurrency 4"

run () {  # name, extra flags
  local name="$1"; shift
  echo "=== $name started $(date +%H:%M:%S) ===" > "$D/$name.log"
  stdbuf -oL -eL .venv/bin/python -u -m bench.run $FLAGS "$@" --run-id "lme-v23-$name" 2>&1 \
    | grep --line-buffered -vE "^AFC|query_embedding_failed" >> "$D/$name.log"
  echo "=== $name DONE $(date +%H:%M:%S) ===" >> "$D/$name.log"
}

run dynkonly --dynamic-k &
run pro      --answer-model gemini-2.5-pro &
run think    --thinking-budget 2048 &
run temporal &
wait
echo "ALL DONE $(date +%H:%M:%S)" > "$D/parallel.done"
