"""Contradiction adjudication: recall past the seven-string negation list.

The lexical pass finds a conflict when two similar memories differ by a
negation from a fixed list, an antonym from a fixed list, or a figure. These
tests are about the cases those lists cannot express, and about the one
property that makes putting a model on the write path safe here: it fails
CLOSED. A broken vendor proposes nothing, which is what this feature did
before a vendor existed.
"""

from __future__ import annotations

import pytest

from mapi.core.ids import new_id
from mapi.domain.consolidation import unexplained_pairs
from mapi.domain.models import Memory, MemoryStatus
from mapi.domain.synthesis.adjudicate import (
    MIN_CONFIDENCE,
    adjudicate_contradictions,
    format_candidates,
    parse_verdicts,
)

CANDIDATES = [("mem_a", "I eat meat"), ("mem_b", "I live in Berlin")]


def test_candidates_are_numbered_not_keyed_by_id() -> None:
    """A transposed character in a ULID would attach a conflict to the wrong row."""
    rendered = format_candidates(CANDIDATES)
    assert rendered == "1. I eat meat\n2. I live in Berlin"
    assert "mem_a" not in rendered


def test_parse_maps_indexes_back_to_ids() -> None:
    verdicts = parse_verdicts(
        '[{"n": 1, "reason": "vegetarian now", "confidence": 0.9}]', CANDIDATES
    )
    assert [v.memory_id for v in verdicts] == ["mem_a"]
    assert verdicts[0].reason == "vegetarian now"


def test_parse_survives_a_code_fence() -> None:
    raw = '```json\n[{"n": 2, "reason": "moved", "confidence": 0.95}]\n```'
    assert [v.memory_id for v in parse_verdicts(raw, CANDIDATES)] == ["mem_b"]


def test_parse_salvages_a_truncated_array() -> None:
    """A cut-off reply must not read as "nothing conflicts"."""
    raw = '[{"n": 1, "reason": "stopped eating meat", "confidence": 0.88}, {"n": 2, "rea'
    assert [v.memory_id for v in parse_verdicts(raw, CANDIDATES)] == ["mem_a"]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "",
        "no conflicts found",
        '[{"n": 99, "confidence": 0.99}]',  # out of range
        '[{"n": 1}]',  # no confidence
        '[{"n": "one", "confidence": 0.9}]',  # not a number
        f'[{{"n": 1, "confidence": {MIN_CONFIDENCE - 0.01}}}]',  # under the floor
    ],
)
def test_doubtful_output_yields_no_conflict(raw: str) -> None:
    assert parse_verdicts(raw, CANDIDATES) == []


def test_a_duplicate_index_is_counted_once() -> None:
    raw = '[{"n": 1, "confidence": 0.9}, {"n": 1, "confidence": 0.95}]'
    assert len(parse_verdicts(raw, CANDIDATES)) == 1


async def test_it_finds_what_the_word_list_cannot() -> None:
    seen: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return '[{"n": 1, "reason": "no longer eats meat", "confidence": 0.92}]'

    verdicts = await adjudicate_contradictions("I no longer eat meat", CANDIDATES, complete)
    assert [v.memory_id for v in verdicts] == ["mem_a"]
    assert "I no longer eat meat" in seen["prompt"]


async def test_a_vendor_failure_proposes_nothing() -> None:
    """Fails CLOSED. A false contradiction costs more than a missed one."""

    async def broken(prompt: str) -> str:
        raise RuntimeError("upstream is down")

    assert await adjudicate_contradictions("anything", CANDIDATES, broken) == []


async def test_a_timeout_proposes_nothing_and_does_not_block() -> None:
    import asyncio

    async def slow(prompt: str) -> str:
        await asyncio.sleep(10)
        return '[{"n": 1, "confidence": 0.99}]'

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await adjudicate_contradictions("x", CANDIDATES, slow, timeout_s=0.01) == []
    assert loop.time() - started < 1.0


async def test_the_candidate_list_is_capped() -> None:
    sent: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        sent["prompt"] = prompt
        return "[]"

    many = [(f"mem_{i}", f"statement number {i}") for i in range(50)]
    await adjudicate_contradictions("x", many, complete, max_candidates=3)
    assert "3. statement number 2" in sent["prompt"]
    assert "4. statement number 3" not in sent["prompt"]


async def test_no_candidates_makes_no_call() -> None:
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "[]"

    assert await adjudicate_contradictions("x", [], complete) == []
    assert not called


# -- the seam: which pairs even reach the model ---------------------------


def _memory(content: str) -> Memory:
    return Memory(org_id=new_id("org"), space_id=new_id("space"), content=content)


def test_pairs_the_lexical_pass_already_explained_are_not_re_asked() -> None:
    """No point paying a model to confirm what a regex already found."""
    new = _memory("I do not eat meat")
    old = _memory("I eat meat")
    vector = [1.0, 0.0, 0.0]
    # Near-identical embeddings put the pair inside the contradiction band.
    close = [0.92, 0.39, 0.0]

    remaining = unexplained_pairs(new, vector, [(old, close)])
    assert remaining == [], "the negation list caught this one already"


def test_a_conflict_outside_the_word_list_is_handed_over() -> None:
    new = _memory("I gave up eating meat")
    old = _memory("I eat meat")
    vector = [1.0, 0.0, 0.0]
    close = [0.92, 0.39, 0.0]

    remaining = unexplained_pairs(new, vector, [(old, close)])
    assert [m.id for m, _ in remaining] == [old.id]


def test_distant_memories_never_reach_the_model() -> None:
    """The similarity gate is what keeps this affordable."""
    new = _memory("I gave up eating meat")
    unrelated = _memory("The API runs on Python")
    assert unexplained_pairs(new, [1.0, 0.0, 0.0], [(unrelated, [0.0, 1.0, 0.0])]) == []


def test_superseded_memories_are_not_adjudicated() -> None:
    new = _memory("I gave up eating meat")
    old = _memory("I eat meat")
    old.status = MemoryStatus.SUPERSEDED
    assert unexplained_pairs(new, [1.0, 0.0, 0.0], [(old, [0.92, 0.39, 0.0])]) == []


# -- lineage: claims never revise the passage they came from ---------------
#
# Extraction turns one paragraph into several claims, each near-identical to
# the parent and to its siblings. Run through the revision checks unfiltered,
# that produced real damage on real data: a claim was flagged as contradicting
# its own parent, and "Krishna is affiliated with Reakon Labs" was applied as
# SUPERSEDING "Principal Engineer and co-founder of Reakon Labs since May
# 2026" -- a vague claim hiding the specific fact it was extracted from.


def _claim(content: str, parent_id: str) -> Memory:
    return Memory(
        org_id=new_id("org"),
        space_id=new_id("space"),
        content=content,
        metadata={"extracted_from": parent_id},
    )


def test_a_claim_shares_lineage_with_its_parent() -> None:
    from mapi.domain.consolidation import same_lineage

    parent = _memory("Krishna is on an F-1 visa and needs CPT during semesters.")
    claim = _claim("Krishna is on an F-1 visa.", parent.id)
    assert same_lineage(claim, parent)
    assert same_lineage(parent, claim), "the relation is symmetric"


def test_two_claims_from_one_passage_share_lineage() -> None:
    from mapi.domain.consolidation import same_lineage

    parent_id = new_id("memory")
    a = _claim("Krishna's GitHub username is kbhatnagar1506.", parent_id)
    b = _claim("Krishna made 1,612 contributions in the past year.", parent_id)
    assert same_lineage(a, b)


def test_unrelated_memories_do_not_share_lineage() -> None:
    from mapi.domain.consolidation import same_lineage

    a = _memory("Krishna is on an F-1 visa.")
    b = _memory("Reakon runs on Next.js 15.")
    assert not same_lineage(a, b)
    # Claims of DIFFERENT parents must still be able to revise each other.
    c = _claim("Krishna lives in Berlin.", new_id("memory"))
    d = _claim("Krishna lives in Madrid.", new_id("memory"))
    assert not same_lineage(c, d)


def test_a_claim_is_never_proposed_as_superseding_its_parent() -> None:
    from mapi.domain.consolidation import propose_supersessions

    parent = _memory("Krishna is Principal Engineer and co-founder of Reakon Labs.")
    claim = _claim("Krishna is affiliated with Reakon Labs.", parent.id)
    vector, close = [1.0, 0.0, 0.0], [0.96, 0.28, 0.0]

    assert propose_supersessions(claim, vector, [(parent, close)]) == []


def test_a_claim_is_never_proposed_as_contradicting_its_parent() -> None:
    from mapi.domain.consolidation import propose_contradictions

    parent = _memory("Krishna is on an F-1 visa and does not hold a green card.")
    claim = _claim("Krishna is on an F-1 visa.", parent.id)
    vector, close = [1.0, 0.0, 0.0], [0.92, 0.39, 0.0]

    assert propose_contradictions(claim, vector, [(parent, close)]) == []


# -- a summary never replaces the fact it summarises ----------------------


@pytest.mark.parametrize(
    ("new", "old"),
    [
        (
            "Krishna is affiliated with Reakon Labs.",
            "Krishna is Principal Engineer and co-founder of Reakon Labs Pvt. Ltd. "
            "since May 2026.",
        ),
        (
            "Krishna's GitHub username is kbhatnagar1506.",
            "Krishna's GitHub account kbhatnagar1506 shows 1,612 contributions in "
            "the past year.",
        ),
    ],
)
def test_a_lossy_summary_is_never_a_replacement(new: str, old: str) -> None:
    """Both of these were applied for real, hiding the specific fact."""
    from mapi.domain.synthesis.adjudicate import _is_less_specific

    assert _is_less_specific(new, old)


@pytest.mark.parametrize(
    ("new", "old"),
    [
        # A genuine revision: same attribute, new value.
        ("I live in Madrid now.", "I live in Berlin"),
        ("The API runs on Cloud Run now.", "The API runs on Heroku with two web dynos."),
        (
            "The standup is now at 10:15am every weekday in the main room.",
            "The standup is at 9:30am every weekday in the main room.",
        ),
        # The specific direction must stay allowed.
        (
            "Krishna is Principal Engineer and co-founder of Reakon Labs since May 2026.",
            "Krishna is affiliated with Reakon Labs.",
        ),
    ],
)
def test_real_revisions_are_not_blocked_as_summaries(new: str, old: str) -> None:
    from mapi.domain.synthesis.adjudicate import _is_less_specific

    assert not _is_less_specific(new, old)


async def test_a_summary_never_reaches_the_adjudicator() -> None:
    """Filtered before the call, so no prompt can talk the guard down."""
    from mapi.domain.synthesis.adjudicate import adjudicate_supersessions

    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return '[{"n": 1, "reason": "restated", "confidence": 0.99}]'

    verdicts = await adjudicate_supersessions(
        "Krishna is affiliated with Reakon Labs.",
        [
            (
                "mem_specific",
                "Krishna is Principal Engineer and co-founder of Reakon Labs "
                "Pvt. Ltd. since May 2026.",
            )
        ],
        complete,
    )
    assert verdicts == []
    assert not called


# -- a supersession must name what changed ---------------------------------
#
# Audited on a 10-session trace: 5 of 9 applied supersessions were wrong, and
# every one was the same shape -- a new statement that CONFIRMED or merely
# MENTIONED the old one, read as replacing it. The clearest was "Redis is
# unchanged at 45" hiding "Redis costs 45 dollars a month".
#
# Prose rules in the prompt did not hold. So the model must now quote both
# values, and the values are checked here: a verdict that cannot name two
# different values is not a replacement.

_PAIR = [("m1", "Redis costs 45 dollars a month."), ("m2", "Deploys on Heroku.")]


def _verdicts(raw: str):
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    return parse_supersede_verdicts(raw, _PAIR)


def test_a_confirmation_is_not_a_replacement() -> None:
    raw = (
        '[{"n":1,"attribute":"monthly cost","old_value":"45 dollars",'
        '"new_value":"45","confidence":0.95}]'
    )
    assert _verdicts(raw) == []


def test_the_same_value_in_another_format_is_not_a_replacement() -> None:
    """ "$45.00" and "45 dollars" are one value wearing two formats."""
    raw = (
        '[{"n":1,"attribute":"cost","old_value":"$45.00",'
        '"new_value":"45 dollars","confidence":0.99}]'
    )
    assert _verdicts(raw) == []


def test_a_verdict_that_names_no_values_is_rejected() -> None:
    """Could not say what changed, so nothing demonstrably changed."""
    raw = '[{"n":1,"attribute":"redis","confidence":0.99}]'
    assert _verdicts(raw) == []


def test_a_genuine_change_is_confirmed_and_explains_itself() -> None:
    raw = (
        '[{"n":1,"attribute":"monthly cost","old_value":"45 dollars",'
        '"new_value":"60 dollars","confidence":0.9}]'
    )
    out = _verdicts(raw)
    assert [v.memory_id for v in out] == ["m1"]
    assert "45" in out[0].reason and "60" in out[0].reason


def test_a_textual_change_with_no_numbers_still_counts() -> None:
    raw = (
        '[{"n":2,"attribute":"host","old_value":"Heroku",'
        '"new_value":"Cloud Run","confidence":0.9}]'
    )
    assert [v.memory_id for v in _verdicts(raw)] == ["m2"]


def test_case_alone_is_not_a_change() -> None:
    raw = (
        '[{"n":2,"attribute":"host","old_value":"Heroku",'
        '"new_value":"heroku","confidence":0.9}]'
    )
    assert _verdicts(raw) == []


# -- a value must be a value -----------------------------------------------
#
# With the attribute/old_value/new_value contract in place, the surviving false
# supersessions on a 10-session trace were two shapes, both visible only once
# the adjudicator's own reason reached the edge:
#
#   echo:        "Redis status: Redis is unchanged at 45 dollars. ->
#                 The user is NOT moving off Redis."
#   non-value:   "Heroku cost: unknown -> 175 dollars a month"
#
# The first pastes whole sentences into both slots, which passes a difference
# check and asserts nothing. The second concedes the old value is not known and
# hides a memory anyway -- it hid the Python-and-dynos stack fact on the
# strength of a cost it could not state.


def test_a_sentence_echoed_into_the_value_slots_is_rejected() -> None:
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    candidates = [("m1", "Redis is unchanged at 45 dollars.")]
    raw = (
        '[{"n":1,"attribute":"Redis status",'
        '"old_value":"Redis is unchanged at 45 dollars.",'
        '"new_value":"The user is NOT moving off Redis.","confidence":0.85}]'
    )
    assert parse_supersede_verdicts(raw, candidates, "The user is NOT moving off Redis.") == []


@pytest.mark.parametrize(
    "placeholder",
    [
        # The six a word list would have named.
        "unknown",
        "n/a",
        "not stated",
        "none",
        "-",
        "",
        # And the ones it would not. These are the point: the rule is that the
        # old value must be readable out of the memory being hidden, so it does
        # not matter which phrasing the model reaches for.
        "TBD",
        "???",
        "not recorded",
        "wasn't said",
        "unstated",
        "omitted",
        "no prior value",
        "the memory does not say",
        "cannot tell",
        "blank",
    ],
)
def test_a_value_absent_from_the_old_memory_is_rejected(placeholder: str) -> None:
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    candidates = [("m1", "The API runs on Heroku with two web dynos, Python 3.13.")]
    raw = (
        f'[{{"n":1,"attribute":"Heroku cost","old_value":"{placeholder}",'
        '"new_value":"175 dollars a month","confidence":0.9}]'
    )
    assert parse_supersede_verdicts(raw, candidates, "Heroku is now 175 a month.") == []


def test_an_old_value_the_memory_never_stated_is_rejected() -> None:
    """The generalisation the word list could not have reached.

    Both statements say Cloud SQL. The model reported the change as "Heroku
    Postgres -> Cloud SQL", which is the history of the migration rather than a
    difference between these two memories -- so the new statement adds a date
    and provenance and replaces nothing. Caught on a live trace, where it had
    been counted as a correct supersession by eye.
    """
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    candidates = [
        (
            "m1",
            "The user's primary database is Postgres 16 running on Cloud SQL in us-central1.",
        )
    ]
    raw = (
        '[{"n":1,"attribute":"primary database provider",'
        '"old_value":"Heroku Postgres","new_value":"Cloud SQL","confidence":1.0}]'
    )
    statement = "The user migrated off Heroku Postgres onto Cloud SQL on 03 March 2026."
    assert parse_supersede_verdicts(raw, candidates, statement) == []


def test_a_new_value_may_be_inferred_rather_than_quoted() -> None:
    """Quotability is required of the OLD value only.

    A revision's new value is often implied rather than stated -- cancelling an
    add-on means it now costs nothing, and the statement says "cancelled", not
    "0 dollars". Requiring both sides to be quotable would reject exactly the
    revisions this feature exists to record.
    """
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    candidates = [("m1", "The user's Redis cost is 45 dollars a month this quarter.")]
    raw = (
        '[{"n":1,"attribute":"Redis cost","old_value":"45 dollars a month this quarter",'
        '"new_value":"0 dollars a month (cancelled)","confidence":0.9}]'
    )
    out = parse_supersede_verdicts(
        raw, candidates, "The user cancelled the unused Heroku Redis add-on."
    )
    assert [v.memory_id for v in out] == ["m1"]


@pytest.mark.parametrize(
    ("old", "source"),
    [
        ("$45.00", "The plan costs $45.00 every month."),
        ("45 dollars", "The plan costs 45 dollars every month."),
        ("two web dynos", "The API runs on Heroku with two web dynos, Python 3.13."),
        ("Python 3.13", "The API runs on Heroku with two web dynos, Python 3.13."),
        ("needs stating earlier", "The threat model in the paper needs stating earlier."),
    ],
)
def test_a_real_quote_is_not_mistaken_for_a_placeholder(old: str, source: str) -> None:
    """Punctuation and formatting must not make a genuine quote unquotable."""
    from mapi.domain.synthesis.adjudicate import _is_quotable

    assert _is_quotable(old, source)


@pytest.mark.parametrize(
    ("attribute", "old", "new"),
    [
        ("cost", "340 dollars a month", "410 dollars a month"),
        ("page count", "9 pages", "14 pages"),
        ("host", "Heroku", "Cloud Run"),
    ],
)
def test_a_named_change_still_passes(attribute: str, old: str, new: str) -> None:
    """The checks must aim supersession, not disable it."""
    from mapi.domain.synthesis.adjudicate import parse_supersede_verdicts

    candidates = [("m1", f"The {attribute} is {old}.")]
    raw = (
        f'[{{"n":1,"attribute":"{attribute}","old_value":"{old}",'
        f'"new_value":"{new}","confidence":0.95}}]'
    )
    out = parse_supersede_verdicts(raw, candidates, f"The {attribute} is now {new}.")
    assert [v.memory_id for v in out] == ["m1"]
