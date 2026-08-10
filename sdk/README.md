# mapi-sdk

Python client for **mapi** — memory for AI agents.

```bash
pip install mapi-sdk
```

## Quick start

```python
from mapi_sdk import Mapi

mapi = Mapi(api_key="sm_...")          # or set MAPI_API_KEY
mapi.get_or_create_space("ada")

mapi.add("Prefers window seats", space="ada")
mapi.add("Allergic to shellfish", space="ada", tags=["health"])

for hit in mapi.search("dietary restrictions", space="ada"):
    print(round(hit.score, 3), hit.content)
```

A **space** is one person's memory. Pass its slug or its id — slugs are
resolved once and cached.

## Why a memory is not just a row

```python
ctx = mapi.context(memory_id, space="ada")

ctx.is_current        # False if something replaced it
ctx.current_head      # what replaced it
ctx.replaced          # what it replaced
ctx.derived_from      # the episodes it was computed from
ctx.contradicts       # what disagrees with it
```

Superseded memories are excluded from `search` by default. They are still
stored and still reachable — returning one next to its replacement is how an
agent states last month's answer with this month's confidence.

## Writing well

```python
mapi.add(
    "Ran the charity 5K in 27:12",
    space="ada",
    occurred_at=datetime(2023, 5, 20, tzinfo=UTC),   # when it HAPPENED
    tags=["running"],
    extract=True,          # also store the atomic claims this states
    detect_conflicts=True, # flag anything it disagrees with
)
```

`occurred_at` is event time, not write time. Leaving it out makes a backfill
look like it all happened today, which breaks every "what order did these
come in" question afterwards.

`extract=True` spends a model call to decompose the text into standalone
claims, stored alongside the original with `derived_from` edges back to it.
Measured on LongMemEval it helps questions about what is *true* and hurts
questions about what *happened*, so it is a choice rather than an upgrade.

## Errors

```python
from mapi_sdk import NotFoundError, RateLimitError

try:
    mapi.search("anything", space="nope")
except NotFoundError as exc:
    print(exc.request_id)   # quote this when reporting a problem
```

429 and gateway errors are retried automatically with jittered backoff.
`500` is not retried: a request that made the server throw will usually throw
again, and retrying hides it.

## Configuration

| argument | default |
|---|---|
| `api_key` | `MAPI_API_KEY` |
| `base_url` | the hosted service |
| `timeout` | 30s |
| `max_retries` | 3 |

MIT licensed.
