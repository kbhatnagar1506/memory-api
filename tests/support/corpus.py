"""Seeded corpora of fictional facts with known gold ids and known geometry.

Three properties make a corpus usable as a measuring instrument, and all three
are easy to get wrong:

**1. Nothing is answerable from parametric knowledge.** Entities are composed
from syllables and checked against a blocklist of real-world tokens. A corpus
about Postgres and Redis lets a reader (or, later, a model) answer from what it
already knows, and then the test is measuring pretraining. U-NIAH calls this the
"Starlight Academy" trick; it matters here even without a model in the loop,
because a real embedder puts known entities in known places.

**2. Gold labels come from the generator, never from a judge.** Every fact
carries a synthetic `gold_id` which is also embedded in its text as a marker, so
`ScriptedEmbedder` can bind a vector to it and the assertion can name it.

**3. Time is relative to a captured instant.** `RetrievalPipeline` calls
`datetime.now()` for recency decay, so a corpus with absolute dates scores
differently as the calendar advances -- a test that passes in August and fails in
November. `Corpus.now` is captured once at build time and every `occurred_at` is
an offset from it. Tests that assert on scores also set `use_decay=False`; this
just removes the second-order version of the same trap.

`load()` writes DIRECTLY via `store.upsert_memory` rather than through
`service.ingest`. Measured: 0.07 ms versus 7.4 ms per write, and more
importantly ingest interposes dedupe, a `neighbours` lookup, supersession,
contradiction detection and association -- five uncontrolled transformations
between the corpus you specified and the one you got. Write-path families call
ingest deliberately and keep their corpora small.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from mapi.domain.embeddings.base import Vector

from .embedder import ScriptedEmbedder
from .factories import memory as build_memory

#: Real-world tokens a generated entity must never contain. Not exhaustive, and
#: not trying to be: it catches the failure mode where a "random" name lands on
#: something a model has read about, which is what silently turns a memory test
#: into a knowledge test.
_REAL_WORLD = frozenset(
    {
        "post",
        "gres",
        "red",
        "is",
        "her",
        "oku",
        "python",
        "java",
        "cloud",
        "amazon",
        "google",
        "apple",
        "meta",
        "open",
        "ai",
        "gpt",
        "claude",
        "sql",
        "http",
        "aws",
        "gcp",
        "azure",
        "linux",
        "unix",
        "win",
    }
)

_ONSETS = ("br", "cl", "dr", "fl", "gl", "kr", "pl", "qu", "sn", "th", "tr", "vr", "zh")
_NUCLEI = ("a", "e", "i", "o", "u", "ae", "ei", "ou", "ya")
_CODAS = ("lk", "mp", "nd", "ng", "rk", "sk", "st", "th", "x", "zz")


def fictional(seed: int, count: int) -> list[str]:
    """`count` pronounceable entity names that mean nothing to anyone.

    Deterministic in `seed`, so a failure is reproducible from the test's own
    parameters with no fixture file to keep in sync.
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        word = rng.choice(_ONSETS) + rng.choice(_NUCLEI) + rng.choice(_CODAS)
        if word in seen or any(real in word for real in _REAL_WORLD):
            continue
        seen.add(word)
        out.append(word.capitalize())
    return out


@dataclass(frozen=True, slots=True)
class Fact:
    """One unit of gold evidence.

    `gold_id` appears in `text` as a marker so the embedder can bind a vector to
    it through headers and chunking, and appears in `metadata["gold_id"]` so an
    assertion can recover it from a retrieved memory without parsing prose.
    """

    gold_id: str
    text: str
    vector: Vector
    days_ago: int = 0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str = ""

    def occurred_at(self, now: datetime) -> datetime:
        return now - timedelta(days=self.days_ago)


@dataclass(frozen=True, slots=True)
class Query:
    """A question with its gold set already known."""

    key: str
    text: str
    vector: Vector
    gold_ids: tuple[str, ...]
    hops: int = 1


@dataclass(slots=True)
class Corpus:
    """A built corpus. `memory_id` fields are filled in by `load`."""

    facts: list[Fact]
    queries: dict[str, Query]
    seed: int
    embedder_pin: str
    now: datetime

    def fact(self, gold_id: str) -> Fact:
        for item in self.facts:
            if item.gold_id == gold_id:
                return item
        raise KeyError(f"no fact {gold_id!r} in corpus")

    def memory_ids(self, gold_ids: Sequence[str]) -> list[str]:
        """Gold ids -> memory ids, for comparing against retrieval output."""
        return [self.fact(g).memory_id for g in gold_ids]

    @property
    def fact_of(self) -> dict[str, str]:
        """memory_id -> gold_id, the mapping `unique_information_at_k` needs."""
        return {f.memory_id: f.gold_id for f in self.facts if f.memory_id}

    def gold_of(self, memory_ids: Sequence[str]) -> list[str]:
        """Retrieved memory ids -> gold ids, unknown ids dropped."""
        lookup = self.fact_of
        return [lookup[mid] for mid in memory_ids if mid in lookup]


def build(
    facts: Sequence[Fact],
    queries: Sequence[Query],
    *,
    seed: int = 0,
    embedder_pin: str = "scripted-v1",
    now: datetime | None = None,
) -> Corpus:
    """Assemble a corpus. `now` is captured once; see the module docstring."""
    gold_ids = [f.gold_id for f in facts]
    if len(set(gold_ids)) != len(gold_ids):
        duplicates = sorted({g for g in gold_ids if gold_ids.count(g) > 1})
        raise ValueError(f"duplicate gold ids: {duplicates}")
    for query in queries:
        unknown = set(query.gold_ids) - set(gold_ids)
        if unknown:
            raise ValueError(f"query {query.key!r} references unknown gold {sorted(unknown)}")
    return Corpus(
        facts=list(facts),
        queries={q.key: q for q in queries},
        seed=seed,
        embedder_pin=embedder_pin,
        now=now or datetime.now(UTC),
    )


def register(corpus: Corpus, embedder: ScriptedEmbedder) -> None:
    """Bind every fact's and query's vector, by marker for facts.

    Facts register by MARKER because their text is wrapped in a context header
    and possibly chunked before it reaches the embedder. Queries register
    EXACTLY because a query string reaches the embedder verbatim.
    """
    for item in corpus.facts:
        embedder.register_marker(item.gold_id, item.vector)
    for query in corpus.queries.values():
        embedder.register(query.text, query.vector)


async def load(store: Any, corpus: Corpus, *, org_id: str, space_id: str) -> Corpus:
    """Write every fact and record the memory id it landed on.

    Insertion order is the order of `corpus.facts`, which is what the
    position-sweep family varies. Returns the corpus so a caller can chain.
    """
    for index, item in enumerate(corpus.facts):
        stored = build_memory(
            org_id=org_id,
            space_id=space_id,
            content=item.text,
            vector=item.vector,
            tags=list(item.tags),
            metadata={**item.metadata, "gold_id": item.gold_id, "insert_order": index},
            occurred_at=item.occurred_at(corpus.now),
        )
        await store.upsert_memory(stored)
        corpus.facts[index] = replace(item, memory_id=stored.id)
    return corpus


__all__ = ["Corpus", "Fact", "Query", "build", "fictional", "load", "register"]
