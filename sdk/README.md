# mapi-sdk

Python client for **mapi** — memory for AI agents.

```bash
pip install mapi-sdk
```

## Quick start

```python
from mapi_sdk import Mapi

client = Mapi(api_key="sm_...")          # or set MAPI_API_KEY
client.spaces.get_or_create("ada")

client.memories.add("Prefers window seats", space="ada", tags=["travel"])

for hit in client.search.execute("seating preference", space="ada"):
    print(round(hit.score, 3), hit.content)
```

Async is the same surface, awaited:

```python
from mapi_sdk import AsyncMapi

async with AsyncMapi() as client:
    await client.memories.add("Allergic to shellfish", space="ada")
    hits = await client.search.execute("allergies", space="ada")
```

A **space** is one person's memory. Pass its slug or its id — slugs resolve
once and cache, so the readable name costs one request per process rather
than one per call.

## Resources

| namespace | what it does |
|---|---|
| `client.memories` | `add`, `add_many`, `get`, `list`, `delete`, `erase`, `context`, `lineage`, `versions`, `relate`, `relations` |
| `client.search` | `execute` |
| `client.spaces` | `list`, `get`, `create`, `get_or_create`, `delete` |
| `client.graph` | `get` |

`client.add(...)` and `client.query(...)` exist as shortcuts for the two calls
that make up most usage.

## Why a memory is not just a row

```python
ctx = client.memories.context(memory_id, space="ada")

ctx.is_current        # False if something replaced it
ctx.current_head      # what replaced it
ctx.replaced          # what it replaced
ctx.derived_from      # the episodes it was computed from
ctx.contradicts       # what disagrees with it
```

Superseded memories are excluded from search by default. They are still
stored and still reachable — returning one beside its replacement is how an
agent states last month's answer with this month's confidence.

## Writing well

```python
from datetime import UTC, datetime

client.memories.add(
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
    client.search.execute("anything", space="nope")
except NotFoundError as exc:
    print(exc.request_id)   # quote this when reporting a problem
```

`AuthenticationError`, `PermissionError_`, `NotFoundError`, `ConflictError`,
`ValidationError`, `RateLimitError`, `ServerError`, `ConnectionError_` — all
subclasses of `MapiError`, all carrying the server's `request_id`.

429 and gateway errors retry automatically with jittered backoff. `500` does
not: a request that made the server throw will usually throw again, and
retrying only hides it.

## Configuration

| argument | default |
|---|---|
| `api_key` | `MAPI_API_KEY` |
| `base_url` | the hosted service |
| `timeout` | 30s |
| `max_retries` | 3 |

The only dependency is `httpx`. Responses are plain dataclasses, each keeping
`.raw`, so a field added server-side is reachable the day it ships.

MIT licensed.
