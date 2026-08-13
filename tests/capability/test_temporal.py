"""Family E: time, which is the capability every benchmark finds weakest.

BEAM found contradiction resolution the worst of its ten abilities.
MemoryAgentBench found ALL published memory systems fail multi-hop conflict
resolution at 6% or less, and its FactConsolidation task *tells the agent* that
newer facts have larger serial numbers -- the recency rule is stated outright and
still not followed, with the strongest system at 54%. LongMemEval measures
knowledge-update as one of five core abilities. Zep's whole contribution is a
bi-temporal model separating when a fact was true from when it was learned.

This system has more machinery for that than most: `occurred_at` (event time)
versus `valid_from`/`valid_to` (system time), a `supersedes` edge, a suppression
stage, and an as-of read. What it did not have is a test that walks the chain and
checks which value comes back.

Two things this family deliberately does NOT do. It does not use recency decay --
that stage multiplies by wall-clock age, so any score assertion under it is
non-hermetic and passes in August and fails in November. And it does not measure
"does the model answer correctly"; it measures which memory is retrievable, which
is the part a deterministic system owes the reader.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.support.factories import edge
from tests.support.factories import memory as build_memory
from tests.support.harness import GEOMETRY
from tests.support.metrics import rank_of
from tests.support.vectors import ANCHOR, at_cosine, axis, nudge

from mapi.domain.models import MemoryStatus, RelationType
from mapi.domain.retrieval.pipeline import SearchRequest

#: A fixed instant. Every timestamp below is an offset from it, so the corpus
#: does not drift as the calendar advances -- the trap `Corpus.now` exists for.
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

#: Revisions of one slot are near-identical in wording, so they sit close
#: together. 0.88 keeps them below the 0.97 near-duplicate ceiling (which would
#: merge them at write time) and above the supersession floor.
REVISION_COSINE = 0.88


async def _chain(geometry: tuple, versions: int, *, slug: str) -> tuple[object, list[object]]:
    """`versions` revisions of one slot, oldest first, joined by SUPERSEDES.

    Written directly to the store rather than through `ingest`, because ingest's
    own consolidation would decide what supersedes what -- and the point here is
    to test retrieval over a KNOWN chain, not to re-test consolidation.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=slug, name=slug)

    written = []
    for i in range(versions):
        marker = f"v{i:02d}"
        vector = nudge(at_cosine(REVISION_COSINE, off=1), 1e-4 * i, off=40 + i)
        embedder.register_marker(marker, vector)
        stored = build_memory(
            org_id=org.id,
            space_id=space_n.id,
            content=f"{marker} the standup is at {9 + i}:30am",
            vector=vector,
            occurred_at=T0 + timedelta(days=7 * i),
            status=MemoryStatus.ACTIVE if i == versions - 1 else MemoryStatus.SUPERSEDED,
        )
        await service.store.upsert_memory(stored)
        written.append(stored)

    # `source_id` is the NEWER memory, per the RelationEdge contract.
    for i in range(1, versions):
        await service.store.create_relation(
            edge(
                org_id=org.id,
                space_id=space_n.id,
                source_id=written[i].id,
                target_id=written[i - 1].id,
                type=RelationType.SUPERSEDES,
            )
        )
    embedder.register("when is the standup", axis(ANCHOR))
    return space_n, written


async def _search(geometry: tuple, space_n: object, **overrides: object) -> list[str]:
    service, org, _space, _embedder = geometry
    response = await service.search(
        SearchRequest(
            query="when is the standup",
            org_id=org.id,
            space_id=space_n.id,  # type: ignore[attr-defined]
            limit=25,
            **{**GEOMETRY, **overrides},  # type: ignore[arg-type]
        )
    )
    return [hit.memory.id for hit in response.results]


# -- E1: the supersession chain -------------------------------------------


@pytest.mark.parametrize("versions", [1, 2, 5, 20])
async def test_the_current_value_is_what_a_present_tense_query_returns(
    geometry: tuple, versions: int
) -> None:
    """The capability every benchmark finds hardest, at K = 1, 2, 5 and 20.

    MemoryAgentBench's strongest system managed 54% on exactly this, with the
    recency rule stated in the prompt. Here it is deterministic: only the head of
    the chain is ACTIVE, and the default filter is ACTIVE-only, so a present-tense
    query can only see the current value.
    """
    space_n, written = await _chain(geometry, versions, slug=f"e1v{versions}")
    returned = await _search(geometry, space_n)
    assert returned == [written[-1].id], (
        f"a {versions}-version chain returned {len(returned)} memories; "
        "only the head should be retrievable"
    )


async def test_a_twenty_deep_chain_hides_all_nineteen_predecessors(
    geometry: tuple,
) -> None:
    """Depth is where a suppression bug would show: a walker with a depth limit,
    or one that only checks the immediate predecessor, would leak the older
    versions at exactly the point nobody tests."""
    space_n, written = await _chain(geometry, 20, slug="e1deep")
    returned = await _search(geometry, space_n)
    superseded = {m.id for m in written[:-1]}
    assert not (set(returned) & superseded), (
        f"{len(set(returned) & superseded)} superseded versions were retrievable"
    )


# -- E2: the history query, inverted --------------------------------------


def _history_filter() -> object:
    """The filter a history query needs, which is more than the flag.

    See `test_include_superseded_alone_is_not_enough` below: `include_superseded`
    skips the SUPPRESSION STAGE, and the store filter is a separate gate that
    still excludes non-ACTIVE rows. Both have to be set, which is what the HTTP
    route does and what a service-level caller has to remember.
    """
    from mapi.store.base import MemoryFilter

    return MemoryFilter(statuses=frozenset({MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED}))


async def test_include_superseded_alone_is_not_enough(geometry: tuple) -> None:
    """A footgun worth pinning: the flag does half the job.

    `SearchRequest.include_superseded` only skips stage 6, the suppression pass.
    The STORE filter (`SearchRequest.filters.statuses`) defaults to ACTIVE-only
    and runs first, so a superseded memory is gone before suppression is reached.
    The HTTP route compensates by adding SUPERSEDED to the statuses it builds --
    but a caller using the service directly, reading a flag named
    "include_superseded", reasonably expects it to include superseded memories and
    gets an unchanged result set with no error.

    Recorded rather than changed: making the flag widen the filter implicitly
    would mean one field quietly rewriting another, and the two gates are
    genuinely different stages. The fix if this is ever revisited is a clearer
    name or a service-level helper.
    """
    space_n, written = await _chain(geometry, 5, slug="e2flag")
    flag_only = await _search(geometry, space_n, include_superseded=True)
    assert flag_only == [written[-1].id], (
        "include_superseded now widens the store filter too; update this test and "
        "the helper below"
    )

    both = await _search(geometry, space_n, include_superseded=True, filters=_history_filter())
    assert len(both) == len(written)


async def test_the_whole_version_set_is_reachable_on_request(geometry: tuple) -> None:
    """The other direction, and the one over-aggressive hiding would break.

    "What was it before?" is a legitimate question, and if supersession destroyed
    history rather than hiding it there would be nothing to return. Bitemporality
    is only useful if the old values survive.
    """
    space_n, written = await _chain(geometry, 5, slug="e2")
    returned = await _search(
        geometry, space_n, include_superseded=True, filters=_history_filter()
    )
    assert set(returned) == {m.id for m in written}, (
        f"history query returned {len(returned)} of {len(written)} versions"
    )


async def test_the_versions_can_be_ordered_by_event_time(geometry: tuple) -> None:
    """Ordering the history needs `occurred_at`, not `created_at`.

    A backfilled chain -- five revisions written in one batch -- has near-identical
    creation times and correct event times. Anything that ordered by write time
    would present the history in an arbitrary order.
    """
    space_n, written = await _chain(geometry, 5, slug="e2order")
    returned = await _search(
        geometry, space_n, include_superseded=True, filters=_history_filter()
    )
    service, org, _space, _embedder = geometry
    fetched = [await service.get_memory(org.id, space_n.id, mid) for mid in returned]
    by_event = sorted(fetched, key=lambda m: m.occurred_at)
    assert [m.id for m in by_event] == [m.id for m in written]


# -- E3: validity intervals and the as-of read ----------------------------


async def test_an_as_of_read_returns_the_snapshot_that_was_current_then(
    geometry: tuple,
) -> None:
    """System time, which is a different question from event time.

    "What did our database SAY on 15 January" is answered from
    `valid_from`/`valid_to`, and it is the question an audit asks. Built by
    version-bumping one memory so the store closes the old snapshot and opens a
    new one, which is the only way the intervals are real rather than asserted.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="e3", name="e3")
    embedder.register_marker("asof", at_cosine(0.9, off=1))

    original = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="asof the standup is at 9:30am",
        vector=at_cosine(0.9, off=1),
        occurred_at=T0,
    )
    await service.store.upsert_memory(original, now=T0)
    await service.store.upsert_memory(
        original.revise(content="asof the standup is at 10:15am", version=2),
        now=T0 + timedelta(days=14),
    )

    early = await service.get_memory_as_of(
        org.id, space_n.id, original.id, T0 + timedelta(days=7)
    )
    late = await service.get_memory_as_of(
        org.id, space_n.id, original.id, T0 + timedelta(days=21)
    )
    assert early is not None and late is not None
    assert "9:30am" in early.content, "the as-of read returned a future value"
    assert "10:15am" in late.content


async def test_the_as_of_boundary_is_half_open(geometry: tuple) -> None:
    """`valid_from <= t < valid_to`.

    A snapshot that closed exactly at `t` was already superseded at `t`, so the
    boundary belongs to the NEW version. Pinned because an off-by-one here returns
    a stale value for exactly one instant, which is the hardest kind of bug to
    notice and the easiest to introduce.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="e3b", name="e3b")
    embedder.register_marker("bound", at_cosine(0.9, off=1))

    cutover = T0 + timedelta(days=14)
    original = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="bound the standup is at 9:30am",
        vector=at_cosine(0.9, off=1),
        occurred_at=T0,
    )
    await service.store.upsert_memory(original, now=T0)
    await service.store.upsert_memory(
        original.revise(content="bound the standup is at 10:15am", version=2), now=cutover
    )

    at_cutover = await service.get_memory_as_of(org.id, space_n.id, original.id, cutover)
    assert at_cutover is not None
    assert "10:15am" in at_cutover.content, (
        "at the cutover instant the OLD value came back -- the interval is being "
        "treated as closed rather than half-open"
    )


async def test_an_as_of_before_the_first_version_is_a_not_found(
    geometry: tuple,
) -> None:
    """Distinct from "returns the earliest". Nothing was true before the first
    write, and inventing a value for that window would be fabrication."""
    from mapi.core.errors import NotFoundError

    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="e3c", name="e3c")
    embedder.register_marker("before", at_cosine(0.9, off=1))
    stored = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="before the standup is at 9:30am",
        vector=at_cosine(0.9, off=1),
    )
    await service.store.upsert_memory(stored, now=T0)

    with pytest.raises(NotFoundError):
        await service.get_memory_as_of(org.id, space_n.id, stored.id, T0 - timedelta(days=1))


# -- E5: tie-breaks must be deterministic ---------------------------------


async def test_two_near_identical_memories_order_the_same_way_every_time(
    geometry: tuple,
) -> None:
    """Ties resolved by float noise are a real and silent bug.

    Two memories 1e-9 apart in cosine, searched five times. The order must not
    change between runs -- and it must not change between backends either, which
    is why every sort in the pipeline carries an id tiebreak.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="e5", name="e5")

    base = at_cosine(0.85, off=1)
    for i, vector in enumerate((base, nudge(base, 1e-9, off=2))):
        embedder.register_marker(f"tie{i}", vector)
        await service.store.upsert_memory(
            build_memory(
                org_id=org.id,
                space_id=space_n.id,
                content=f"tie{i} the standup detail",
                vector=vector,
                occurred_at=T0 + timedelta(days=i),
            )
        )
    embedder.register("when is the standup", axis(ANCHOR))

    orders = [tuple(await _search(geometry, space_n)) for _ in range(5)]
    assert len(set(orders)) == 1, f"tie order varied across runs: {set(orders)}"


# -- E6: the staleness audit ----------------------------------------------


async def test_an_answer_changes_only_when_something_superseded_it(
    geometry: tuple,
) -> None:
    """Index drift, and non-idempotent updates.

    Ingest a stream and re-run a fixed query at checkpoints. The retrieved head
    may change ONLY when a superseding memory arrived -- an unrelated write must
    not move it. StreamingQA's framing, and the cheapest possible test for a class
    of bug that otherwise surfaces as "the answer changed and nobody knows why".
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="e6", name="e6")
    embedder.register("when is the standup", axis(ANCHOR))

    head = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="s0 the standup is at 9:30am",
        vector=at_cosine(0.90, off=1),
        occurred_at=T0,
    )
    embedder.register_marker("s0", at_cosine(0.90, off=1))
    await service.store.upsert_memory(head)
    first = await _search(geometry, space_n)
    assert first[:1] == [head.id]

    # An unrelated write. The answer must not move.
    embedder.register_marker("u1", at_cosine(0.25, off=5))
    await service.store.upsert_memory(
        build_memory(
            org_id=org.id,
            space_id=space_n.id,
            content="u1 the office lease renews in June",
            vector=at_cosine(0.25, off=5),
            occurred_at=T0 + timedelta(days=1),
        )
    )
    after_unrelated = await _search(geometry, space_n)
    assert after_unrelated[:1] == [head.id], (
        "an unrelated write changed which memory answered the query"
    )

    # A superseding write. Now it must move.
    replacement_vector = nudge(at_cosine(0.90, off=1), 1e-4, off=6)
    embedder.register_marker("s1", replacement_vector)
    replacement = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="s1 the standup is at 10:15am",
        vector=replacement_vector,
        occurred_at=T0 + timedelta(days=14),
    )
    await service.store.upsert_memory(replacement)
    await service.store.upsert_memory(
        head.model_copy(update={"status": MemoryStatus.SUPERSEDED, "version": 2})
    )
    await service.store.create_relation(
        edge(
            org_id=org.id,
            space_id=space_n.id,
            source_id=replacement.id,
            target_id=head.id,
            type=RelationType.SUPERSEDES,
        )
    )
    after_supersession = await _search(geometry, space_n)
    assert after_supersession[:1] == [replacement.id], (
        "a superseding write did not change the answer"
    )
    assert head.id not in after_supersession


async def test_repeating_a_search_is_idempotent(geometry: tuple) -> None:
    """No hidden state. Same corpus, same query, same answer -- five times.

    An access-count boost or a cache that mutated on read would show up here, and
    `decay.access_factor` exists in the tree (unwired) for exactly that idea.
    """
    space_n, written = await _chain(geometry, 3, slug="e6idem")
    results = [tuple(await _search(geometry, space_n)) for _ in range(5)]
    assert len(set(results)) == 1
    assert results[0] == (written[-1].id,)


async def test_supersession_hides_without_destroying(geometry: tuple) -> None:
    """The invariant the whole family rests on.

    Superseded memories are absent from a default search AND present in the
    store. Anything that deleted them instead would make the history query above
    impossible and erasure unauditable.
    """
    space_n, written = await _chain(geometry, 4, slug="e6keep")
    service, org, _space, _embedder = geometry

    default = await _search(geometry, space_n)
    assert set(default) == {written[-1].id}
    for older in written[:-1]:
        fetched = await service.get_memory(org.id, space_n.id, older.id)
        assert fetched is not None, "a superseded memory was destroyed, not hidden"
        assert fetched.status is MemoryStatus.SUPERSEDED
        assert rank_of(default, older.id) is None
