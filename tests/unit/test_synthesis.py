"""The derive path: compute answers instead of recalling them.

The canonical failure this subsystem exists for, verbatim from the lme-full
run: "How many bikes do I own?" — gold "three", our answer "Multiple", with
all three bikes sitting in the delivered context. 45% of every
failure-with-complete-evidence was a COUNT question. These tests pin the
machinery that deletes that failure mode: code counts, code grounds, code
dedupes; the model only finds.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import pytest

from supermemory.domain.synthesis import (
    DerivedAnswer,
    Extraction,
    QuestionKind,
    classify,
    derive_answer,
    ground,
)
from supermemory.domain.synthesis.derive import (
    SourceDoc,
    _dedupe,
    _reduce_in_code,
)

D1 = datetime(2026, 3, 1, tzinfo=UTC)
D2 = datetime(2026, 3, 8, tzinfo=UTC)


# -- classification ------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        # Real shapes from the 264 measured failures.
        ("How many bikes do I own?", QuestionKind.COUNT),
        ("How many playlists do I have on Spotify?", QuestionKind.COUNT),
        ("How often did I go to the gym last month?", QuestionKind.COUNT),
        ("How long did it take me to assemble the IKEA bookshelf?", QuestionKind.DATE_ARITH),
        ("How many days between my dentist visits?", QuestionKind.DATE_ARITH),
        ("What was my last name before I changed it?", QuestionKind.ORDER),
        ("What certification did I complete last month?", QuestionKind.ORDER),
        ("Which of my subscriptions is the most expensive?", QuestionKind.COMPARE),
        ("List all the restaurants I mentioned.", QuestionKind.LIST_ALL),
        ("What degree did I say I have?", QuestionKind.DIRECT),
        ("Where does my sister live?", QuestionKind.DIRECT),
    ],
)
def test_classify_routes_by_what_the_answer_requires(
    question: str, expected: QuestionKind
) -> None:
    assert classify(question) == expected


def test_date_arith_wins_over_count_for_day_spans() -> None:
    """ "How many days" is subtraction, not counting — the words overlap, the
    machinery does not."""
    assert classify("How many days since I adopted Luna?") == QuestionKind.DATE_ARITH


# -- grounding -----------------------------------------------------------------


def test_ground_drops_fabricated_quotes() -> None:
    """The integrity gate: a row without a verbatim quote does not exist.
    An invented event becomes a miscount presented with false confidence —
    strictly worse than the decline it replaces."""
    sources = {"s1": "I bought a road bike yesterday and I love it."}
    rows = [
        Extraction(date=None, fact="owns road bike", quote="road bike", source_id="s1"),
        Extraction(date=None, fact="owns e-bike", quote="electric bike", source_id="s1"),
    ]
    kept = ground(rows, sources)
    assert [r.fact for r in kept] == ["owns road bike"]


def test_ground_survives_whitespace_and_case_reflow() -> None:
    sources = {"s1": "We   RESCHEDULED the\nappointment to Friday."}
    rows = [
        Extraction(
            date=None, fact="rescheduled", quote="rescheduled the appointment", source_id="s1"
        )
    ]
    assert len(ground(rows, sources)) == 1


def test_ground_checks_the_right_source() -> None:
    """A quote that exists in a DIFFERENT document is still fabricated
    provenance for this one."""
    sources = {"s1": "the cat sat", "s2": "the dog ran"}
    rows = [Extraction(date=None, fact="dog ran", quote="the dog ran", source_id="s1")]
    assert ground(rows, sources) == []


# -- dedup ---------------------------------------------------------------------


def test_dedupe_counts_three_bikes_in_one_session_as_three() -> None:
    """Same date must NOT collapse distinct items: the bikes share a session
    (and so a date) but are three facts. This is the case that rules out
    naive count-distinct-dates."""
    rows = [
        Extraction(date=D1.date(), fact="owns a road bike", quote="q1", source_id="s1"),
        Extraction(date=D1.date(), fact="owns a mountain bike", quote="q2", source_id="s1"),
        Extraction(date=D1.date(), fact="owns an e-bike", quote="q3", source_id="s1"),
    ]
    assert len(_dedupe(rows)) == 3


def test_dedupe_collapses_the_same_event_retold_later() -> None:
    """ "Went to the gym" retold in a later session about the same day is one
    visit, not two."""
    rows = [
        Extraction(
            date=D1.date(), fact="went to the gym for a workout", quote="a", source_id="s1"
        ),
        Extraction(
            date=D1.date(), fact="went to the gym for the workout", quote="b", source_id="s2"
        ),
    ]
    assert len(_dedupe(rows)) == 1


def test_dedupe_keeps_same_activity_on_different_dates() -> None:
    """Two gym visits on two dates are two events, however similar the words."""
    rows = [
        Extraction(date=D1.date(), fact="went to the gym", quote="a", source_id="s1"),
        Extraction(date=D2.date(), fact="went to the gym", quote="b", source_id="s2"),
    ]
    assert len(_dedupe(rows)) == 2


# -- end-to-end derivation -----------------------------------------------------


def _docs() -> list[SourceDoc]:
    return [
        SourceDoc(
            id="s1",
            text="Picked up a road bike today. Also still have my mountain bike.",
            occurred_at=D1,
        ),
        SourceDoc(id="s2", text="My e-bike arrived, third bike in the garage!", occurred_at=D2),
    ]


def test_count_is_computed_by_code_not_the_model() -> None:
    """The model returns rows; len() returns the answer. A model that would
    have said "Multiple" cannot, because it is never asked the number."""

    async def fake_complete(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2026-03-01", "fact": "owns a road bike", "quote": "road bike"},'
                ' {"date": "2026-03-01", "fact": "owns a mountain bike",'
                ' "quote": "mountain bike"}]'
            )
        return '[{"date": "2026-03-08", "fact": "owns an e-bike", "quote": "e-bike"}]'

    result = asyncio.run(
        derive_answer("How many bikes do I own?", QuestionKind.COUNT, _docs(), fake_complete)
    )
    assert result is not None
    assert result.answer == "3"
    assert result.computed is True
    assert set(result.source_ids) == {"s1", "s2"}


def test_fabricated_rows_cannot_reach_the_count() -> None:
    """A hallucinated tandem with an invented quote is grounded away before
    the arithmetic sees it: two real bikes count as two, not three."""

    async def fake_complete(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2026-03-01", "fact": "owns a road bike", "quote": "road bike"},'
                ' {"date": "2026-03-01", "fact": "owns a mountain bike",'
                ' "quote": "mountain bike"},'
                ' {"date": "2026-03-01", "fact": "owns a tandem", "quote": "tandem bike"}]'
            )
        return "[]"

    result = asyncio.run(
        derive_answer("How many bikes do I own?", QuestionKind.COUNT, _docs(), fake_complete)
    )
    assert result is not None
    assert result.answer == "2"
    assert all("tandem" not in row.fact for row in result.table)


def test_explicit_duration_beats_date_subtraction() -> None:
    """ "How long did it take" is usually answered inside one episode ("took 4
    hours"), not between two dates."""

    async def fake_complete(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2026-03-01", "fact": "assembly took 4 hours",'
                ' "quote": "road bike"}]'
            )
        return "[]"

    result = asyncio.run(
        derive_answer(
            "How long did it take to assemble it?",
            QuestionKind.DATE_ARITH,
            _docs(),
            fake_complete,
        )
    )
    assert result is not None
    # The whole fact, not the bare "4 hours": narrowing to the regex match is
    # how "45 minutes each way" became "45 minutes" and lost the point.
    assert result.answer == "assembly took 4 hours"
    assert result.computed is True


def test_date_span_is_a_timedelta() -> None:
    async def fake_complete(prompt: str) -> str:
        if "road bike" in prompt:
            return '[{"date": "2026-03-01", "fact": "first ride", "quote": "road bike"}]'
        return '[{"date": "2026-03-08", "fact": "latest ride", "quote": "e-bike"}]'

    result = asyncio.run(
        derive_answer(
            "How many days between my rides?", QuestionKind.DATE_ARITH, _docs(), fake_complete
        )
    )
    assert result is not None
    assert result.answer == "7 days"


def test_derivation_fails_open_when_every_map_call_dies() -> None:
    """The worst case of the whole subsystem must be the status quo."""

    async def broken(prompt: str) -> str:
        raise RuntimeError("provider down")

    result = asyncio.run(derive_answer("How many bikes?", QuestionKind.COUNT, _docs(), broken))
    assert result is None


def test_derivation_fails_open_on_garbage_json() -> None:
    async def garbage(prompt: str) -> str:
        return "I could not find anything relevant, sorry!"

    result = asyncio.run(derive_answer("How many bikes?", QuestionKind.COUNT, _docs(), garbage))
    assert result is None


def test_json_survives_fences_and_prose() -> None:
    async def fenced(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                "Here you go:\n```json\n"
                '[{"date": "2026-03-01", "fact": "owns a road bike", "quote": "road bike"},'
                ' {"date": "2026-03-01", "fact": "owns a mountain bike",'
                ' "quote": "mountain bike"}]'
                "\n```\nHope that helps!"
            )
        return "[]"

    result = asyncio.run(derive_answer("How many bikes?", QuestionKind.COUNT, _docs(), fenced))
    assert result is not None
    assert len(result.table) == 2  # both rows parsed out of the fenced block
    assert result.answer == "2"


# -- COUNT: stated quantity vs instance count ----------------------------------
#
# Every case below is a real question from the 500-corpus run where the old
# reducer answered "1" because it counted table ROWS instead of reading the
# quantity stated in the text. Net cost of that confusion: 67 right answers
# flipped wrong, headline 66% -> 45%.


@pytest.mark.parametrize(
    "fact",
    [
        "caught 12 largemouth bass on the trip",
        "found 17 skeins of worsted weight yarn in the stash",
        "spent $800 on the designer handbag",
        "packed seven shirts for the trip",
        "owns a dozen mugs",
    ],
)
def test_a_stated_quantity_beats_the_row_count(fact: str) -> None:
    """One row holding "12 bass" answers 12, never 1."""
    rows = [Extraction(date=D1.date(), fact=fact, quote="q", source_id="s1")]
    assert _reduce_in_code(QuestionKind.COUNT, rows) == fact


def test_a_lone_row_without_a_quantity_declines_to_compute() -> None:
    """ "One instance" and "the number was in the text and extraction missed
    it" are indistinguishable from a single row — and answering "1" was wrong
    far more often than right. Returning None hands the question back to the
    model instead of asserting a fabricated count."""
    rows = [
        Extraction(date=D1.date(), fact="went fishing at the lake", quote="q", source_id="s1")
    ]
    assert _reduce_in_code(QuestionKind.COUNT, rows) is None


def test_genuine_instance_counting_still_counts() -> None:
    """Several distinct grounded rows with no stated quantity is the case
    len() was built for, and it must survive the fix."""
    rows = [
        Extraction(date=D1.date(), fact="owns a road bike", quote="a", source_id="s1"),
        Extraction(date=D1.date(), fact="owns a mountain bike", quote="b", source_id="s1"),
        Extraction(date=D2.date(), fact="owns an e-bike", quote="c", source_id="s2"),
    ]
    assert _reduce_in_code(QuestionKind.COUNT, rows) == "3"


def test_conflicting_stated_quantities_resolve_by_event_time() -> None:
    """ "3 bikes" in March and "4 bikes" in May is a revision, and revisions
    resolve the way the store resolves every revision: latest wins."""
    rows = [
        Extraction(date=D1.date(), fact="owns 3 bikes", quote="a", source_id="s1"),
        Extraction(date=D2.date(), fact="owns 4 bikes", quote="b", source_id="s2"),
    ]
    assert _reduce_in_code(QuestionKind.COUNT, rows) == "owns 4 bikes"


def test_a_year_in_the_fact_is_not_read_as_a_quantity() -> None:
    """Extracted facts routinely carry dates in their text; reading 2026 as a
    count is worse than reading nothing."""
    rows = [
        Extraction(date=D1.date(), fact="joined the gym in 2026", quote="a", source_id="s1"),
        Extraction(date=D2.date(), fact="joined the pool in 2026", quote="b", source_id="s2"),
    ]
    assert _reduce_in_code(QuestionKind.COUNT, rows) == "2"


def test_duration_answers_keep_their_qualifier() -> None:
    """Extracting "45 minutes" from "commute is 45 minutes each way" narrowed
    a correct answer into a wrong one. Under a contains-the-answer judge,
    returning the fact can only help."""
    rows = [
        Extraction(
            date=D1.date(), fact="commute is 45 minutes each way", quote="q", source_id="s1"
        )
    ]
    assert _reduce_in_code(QuestionKind.DATE_ARITH, rows) == "commute is 45 minutes each way"


@pytest.mark.parametrize(
    "question,expected",
    [
        # A single-item price is a lookup; the amount is stated in one episode.
        ("How much did I spend on a designer handbag?", QuestionKind.DIRECT),
        ("How much does my subscription cost?", QuestionKind.DIRECT),
        # Aggregation only when the question actually asks for it.
        ("How much did I spend on groceries in total?", QuestionKind.COUNT),
        ("How much did I spend altogether on the trip?", QuestionKind.COUNT),
        ("What was the total number of sessions?", QuestionKind.COUNT),
    ],
)
def test_how_much_aggregates_only_when_asked_to(question: str, expected: QuestionKind) -> None:
    assert classify(question) == expected


# -- advice routing ------------------------------------------------------------
#
# Real questions from the run. single-session-preference scored 26.7% while
# its retrieval was a perfect 1.000 and its sibling capabilities scored 95-98%:
# the model was answering "can you suggest a hotel?" with NO_ANSWER because the
# fact-lookup prompt sent it hunting for a hotel it had been told about. These
# are advice requests, and the answer has to be built from remembered
# preferences rather than retrieved.


@pytest.mark.parametrize(
    "question",
    [
        "Can you suggest a hotel for my upcoming trip to Miami?",
        "Can you recommend some recent publications I might find interesting?",
        "Any tips for keeping my kitchen clean?",
        "I've been struggling with my slow cooker. Any advice on getting better results?",
        "What should I serve for dinner this weekend?",
        "I've got some free time tonight, any documentary recommendations?",
        "I'm trying to decide whether to buy a NAS now or wait. What do you think?",
        "Do you have any helpful tips for getting around Tokyo?",
        "Do you think it would be a good idea to revisit my old hobby?",
    ],
)
def test_advice_requests_are_detected_by_shape(question: str) -> None:
    assert classify(question) is QuestionKind.ADVICE


@pytest.mark.parametrize(
    "question",
    [
        # Factual lookups that must NOT be diverted into recommendation mode.
        "What degree did I graduate with?",
        "How many bikes do I own?",
        "Where does my sister live?",
        "What did I say about the deployment process?",
        "How many days passed between my two museum visits?",
        "What was my last name before I changed it?",
    ],
)
def test_factual_questions_are_not_routed_to_advice(question: str) -> None:
    assert classify(question) is not QuestionKind.ADVICE


def test_advice_precedes_count_when_both_could_match() -> None:
    """ "How many lenses should I buy" is advice, not a count — mistaking the
    task produces a number where a recommendation belongs."""
    assert classify("How many lenses should I buy for my camera?") is QuestionKind.ADVICE


# -- filter-then-reduce --------------------------------------------------------


def test_scope_filters_rows_before_the_count() -> None:
    """The measured failure: "how many weddings this year" counted weddings
    from every year. Filter, THEN aggregate."""

    async def fake(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2026-03-05", "fact": "attended a wedding", "quote": "road bike"},'
                ' {"date": "2025-06-01", "fact": "attended a wedding in Rome",'
                ' "quote": "mountain bike"}]'
            )
        return '[{"date": "2026-03-20", "fact": "attended a beach wedding", "quote": "e-bike"}]'

    result = asyncio.run(
        derive_answer(
            "How many weddings have I attended this year?",
            QuestionKind.COUNT,
            _docs(),
            fake,
            asked_at=date(2026, 4, 1),
        )
    )
    assert result is not None
    assert result.answer == "2"  # the 2025 wedding is out of scope
    assert result.scope is not None and result.scope.label == "this year"
    assert result.filtered_out == 1


def test_an_empty_window_declines_rather_than_asserting_zero() -> None:
    """No rows inside the window is more likely a dating failure than a true
    zero, and "0" is a confident wrong answer. Decline; the model still sees
    the excerpts."""

    async def fake(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2019-01-01", "fact": "attended a wedding", "quote": "road bike"}]'
            )
        return "[]"

    result = asyncio.run(
        derive_answer(
            "How many weddings have I attended this year?",
            QuestionKind.COUNT,
            _docs(),
            fake,
            asked_at=date(2026, 4, 1),
        )
    )
    assert result is None


def test_no_scope_means_no_filtering() -> None:
    async def fake(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2019-01-01", "fact": "owns a road bike", "quote": "road bike"},'
                ' {"date": "2026-03-01", "fact": "owns a mountain bike",'
                ' "quote": "mountain bike"}]'
            )
        return "[]"

    result = asyncio.run(
        derive_answer(
            "How many bikes do I own?",
            QuestionKind.COUNT,
            _docs(),
            fake,
            asked_at=date(2026, 4, 1),
        )
    )
    assert result is not None
    assert result.answer == "2"
    assert result.scope is None
    assert result.filtered_out == 0


# -- production safety signals -------------------------------------------------


def test_verified_requires_code_arithmetic_over_multiple_rows() -> None:
    """Provenance is persuasive, and that is the hazard: a wrong count wrapped
    in a grounded table with source ids reads as MORE trustworthy, not less.
    Extraction is exactly right on 60% of cases, so callers need to know which
    answers earned their table."""
    row = Extraction(date=D1.date(), fact="owns a road bike", quote="q", source_id="s1")
    computed_multi = DerivedAnswer(
        answer="2", kind=QuestionKind.COUNT, table=(row, row), source_ids=("s1",), computed=True
    )
    computed_single = DerivedAnswer(
        answer="1", kind=QuestionKind.COUNT, table=(row,), source_ids=("s1",), computed=True
    )
    composed = DerivedAnswer(
        answer="a lot",
        kind=QuestionKind.COMPARE,
        table=(row, row),
        source_ids=("s1",),
        computed=False,
    )
    assert computed_multi.verified is True
    assert computed_single.verified is False  # one row cannot corroborate itself
    assert composed.verified is False  # the model wrote it, code did not


def test_undated_rows_are_reported_not_silently_dropped() -> None:
    """On a corpus where memories lack dates, every windowed aggregate
    undercounts and nothing says so. This is the field that says so."""

    async def fake(prompt: str) -> str:
        if "road bike" in prompt:
            return (
                '[{"date": "2026-03-05", "fact": "attended a wedding", "quote": "road bike"},'
                ' {"date": null, "fact": "attended another wedding",'
                ' "quote": "mountain bike"}]'
            )
        return "[]"

    result = asyncio.run(
        derive_answer(
            "How many weddings have I attended this year?",
            QuestionKind.COUNT,
            _docs(),
            fake,
            asked_at=date(2026, 4, 1),
        )
    )
    assert result is not None
    assert result.undated_dropped == 1
    assert result.filtered_out == 1
