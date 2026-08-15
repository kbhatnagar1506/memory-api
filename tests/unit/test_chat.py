"""Grounded chat: the module with the most stated promises and no tests.

`domain/synthesis/chat.py` was imported by nothing in the suite and its route
(`POST /v1/spaces/{id}/chat`) had never been hit. Its docstring makes three
claims -- it cites, it declines, it carries confidence through -- and each is the
kind of claim that fails silently. An answer that invents a fact is
indistinguishable from one that recalled it unless the citations are checked, and
nothing was checking them.

One defect found and fixed while writing this: `_CITATION` was `\\[(\\d{1,2})\\]`,
so `[100]` did not match. Harmless while the coverage window was capped at 32 by
a hydration bug; the moment that cap was lifted to 200, memories 100 and up
became UNCITABLE -- the model writes `[137]`, the regex misses, and the claim is
reported as uncited, which is the same signal as "the memories did not support
this". A fix in one file made a latent bug in another reachable, which is the
argument for pinning both.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.support.factories import EPOCH
from tests.support.factories import memory as build_memory

from mapi.domain.models import ScoredMemory
from mapi.domain.synthesis.chat import (
    ADVICE_CHAT_PROMPT,
    CHAT_PROMPT,
    MAX_HISTORY_TURNS,
    MAX_MEMORY_CHARS,
    TRUNCATION_MARKER,
    ChatAnswer,
    Turn,
    answer,
    format_history,
    format_memories,
    parse_citations,
)

ORG, SPACE = "org_01k000000000000000000000", "spc_01k000000000000000000000"


def _hit(content: str, *, score: float = 1.0, **kwargs: object) -> ScoredMemory:
    return ScoredMemory(
        memory=build_memory(org_id=ORG, space_id=SPACE, content=content, **kwargs),  # type: ignore[arg-type]
        score=score,
    )


def _replying(text: str):
    async def complete(prompt: str) -> str:
        return text

    return complete


def _capturing(store: dict[str, str], text: str = "ok [1]"):
    async def complete(prompt: str) -> str:
        store["prompt"] = prompt
        return text

    return complete


# -- formatting memories ---------------------------------------------------


def test_memories_are_numbered_from_one() -> None:
    """Numbered, not keyed by id: a 30-character ULID in a citation is thirty
    chances to transpose into an id that exists and attach a real claim to the
    wrong memory."""
    rendered = format_memories([_hit("first fact"), _hit("second fact")])
    assert rendered.startswith("1.")
    assert "\n2." in rendered
    assert ORG not in rendered


def test_the_date_shown_is_event_time_not_write_time() -> None:
    """ "Joined in May 2026" tells the model when the fact became true. The date
    it was typed in tells it nothing it can reason with."""
    rendered = format_memories([_hit("a dated fact", occurred_at=EPOCH)])
    assert "Jan 2026" in rendered


def test_the_date_carries_the_day_not_just_the_month() -> None:
    """Month precision made every sub-month question unanswerable.

    Found live: two memories dated 4 and 24 March both rendered "(Mar 2026)",
    and "how many days did the project take" got "I don't have that in memory"
    -- a correct answer to the context it was actually given. Same-month
    ordering and "what happened on the 12th" failed the same way.
    """
    a = _hit("started", occurred_at=datetime(2026, 3, 4, tzinfo=UTC))
    b = _hit("shipped", occurred_at=datetime(2026, 3, 24, tzinfo=UTC))
    rendered = format_memories([a, b])
    assert "04 Mar 2026" in rendered
    assert "24 Mar 2026" in rendered


def test_a_long_memory_is_clipped() -> None:
    rendered = format_memories([_hit("x" * (MAX_MEMORY_CHARS + 500))])
    assert len(rendered) < MAX_MEMORY_CHARS + 100


def test_a_clipped_memory_says_so() -> None:
    """A truncation the reader cannot see turns "some of this is missing" into
    "this is all there is", and the model answers confidently from a fragment.

    Measured at the old 700-char cap: a 1042-character memory lost its last
    third mid-word, taking the operative clause with it, and nothing in the
    rendered prompt indicated a cut had happened.
    """
    rendered = format_memories([_hit("x" * (MAX_MEMORY_CHARS + 500))])
    assert TRUNCATION_MARKER.strip() in rendered


def test_a_memory_that_fits_is_not_marked() -> None:
    rendered = format_memories([_hit("a short fact")])
    assert TRUNCATION_MARKER.strip() not in rendered
    assert rendered.endswith("a short fact")


@pytest.mark.parametrize("flag", ["unverified", "UNVERIFIED", "low", "unsure", "Low"])
def test_an_unverified_memory_is_labelled(flag: str) -> None:
    """The third promise: a memory system that launders "someone told me this
    once" into a confident answer is actively harmful."""
    rendered = format_memories([_hit("a shaky number", metadata={"confidence": flag})])
    assert "[UNVERIFIED]" in rendered


@pytest.mark.parametrize("value", ["high", "verified", "", None, 0.4, True])
def test_anything_else_is_not_labelled_unverified(value: object) -> None:
    """Including a FLOAT confidence, which is the shape a caller would guess.

    `_is_unverified` only recognises strings, so `{"confidence": 0.1}` is not
    flagged. Pinned as the current contract rather than silently relied on -- a
    caller sending a float gets no label, and should know that.
    """
    rendered = format_memories([_hit("a fact", metadata={"confidence": value})])
    assert "[UNVERIFIED]" not in rendered


def test_no_memories_renders_empty() -> None:
    assert format_memories([]) == ""


# -- formatting history ----------------------------------------------------


def test_history_is_bounded() -> None:
    """An unbounded transcript would crowd out the retrieval it exists to
    support, which inverts the point of the endpoint."""
    turns = [Turn(role="user", content=f"turn {i}") for i in range(MAX_HISTORY_TURNS + 5)]
    rendered = format_history(turns)
    assert rendered.count("User:") == MAX_HISTORY_TURNS
    assert "turn 0" not in rendered


def test_history_keeps_the_most_recent_turns() -> None:
    turns = [Turn(role="user", content=f"turn {i}") for i in range(MAX_HISTORY_TURNS + 2)]
    rendered = format_history(turns)
    assert f"turn {MAX_HISTORY_TURNS + 1}" in rendered


def test_empty_history_renders_nothing() -> None:
    """So the prompt has no dangling "Conversation so far:" header."""
    assert format_history([]) == ""


def test_roles_are_capitalized_for_the_prompt() -> None:
    rendered = format_history([Turn(role="assistant", content="hello")])
    assert "Assistant: hello" in rendered


# -- citations -------------------------------------------------------------


def test_citations_map_to_memory_ids_in_order() -> None:
    hits = [_hit("a"), _hit("b"), _hit("c")]
    cited = parse_citations("per [3] and [1], yes", hits)
    assert cited == [hits[2].memory.id, hits[0].memory.id]


def test_a_repeated_citation_is_counted_once() -> None:
    hits = [_hit("a"), _hit("b")]
    assert parse_citations("[1] and again [1]", hits) == [hits[0].memory.id]


def test_an_out_of_range_citation_is_dropped_not_clamped() -> None:
    """A citation to [9] over eight memories is the model inventing provenance.

    Clamping it to the eighth would manufacture exactly the audit trail this
    function exists to provide.
    """
    hits = [_hit("a"), _hit("b")]
    assert parse_citations("as [9] shows", hits) == []


def test_a_zero_citation_is_dropped() -> None:
    hits = [_hit("a")]
    assert parse_citations("see [0]", hits) == []


def test_a_three_digit_citation_is_matched() -> None:
    """The defect this file found.

    `\\d{1,2}` could not match `[100]`. Latent while coverage was capped at 32;
    reachable the moment that cap was lifted to 200, at which point every memory
    past the 99th was uncitable and any claim resting on one read as unsupported.
    """
    hits = [_hit(f"fact number {i}") for i in range(120)]
    assert parse_citations("per [100]", hits) == [hits[99].memory.id]
    assert parse_citations("per [120]", hits) == [hits[119].memory.id]


def test_a_citation_past_the_three_digit_bound_is_ignored() -> None:
    """`\\d+` would match a year in `[2026]` or a bracketed figure quoted back
    out of the memory text. Bounded, and out-of-range is dropped anyway."""
    hits = [_hit("a")]
    assert parse_citations("in [2026] we shipped", hits) == []


def test_no_citations_yields_an_empty_list() -> None:
    """Which is the shape of an honest refusal, not an error."""
    assert parse_citations("I don't have that in memory.", [_hit("a")]) == []


# -- answering -------------------------------------------------------------


async def test_with_no_memories_the_model_is_never_called() -> None:
    """The second promise. There is nothing to ground an answer in, and asking
    anyway invites exactly the invention rule 1 forbids."""
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "I definitely know this"

    result = await answer("what is my database?", [], complete)
    assert not called
    assert "don't have" in result.reply
    assert result.cited == []
    assert result.considered == []


async def test_the_prompt_carries_the_rules_the_memories_and_the_question() -> None:
    seen: dict[str, str] = {}
    hits = [_hit("Postgres 16 on Cloud SQL")]
    await answer("what database?", hits, _capturing(seen))
    prompt = seen["prompt"]
    assert "Postgres 16 on Cloud SQL" in prompt
    assert "what database?" in prompt
    assert "I don't have that in memory" in prompt, "the decline instruction must survive"
    assert "1." in prompt


#: The routing added for advice is the change in this module most able to break
#: something it was not aimed at. `ADVICE_CHAT_PROMPT` deliberately DROPS rule 1
#: -- decline when the memories do not contain the answer -- because an advice
#: request has no stored answer by construction. That is correct for advice and
#: would be a licence to invent on a fact question, so the branch is pinned from
#: both sides: the advice frame must reach the advice contract, and everything
#: else must still carry the decline instruction.
ADVICE_SHAPED = [
    "Can you suggest a restaurant for Friday?",
    "What laptop should I buy?",
    "Recommend me a book for the flight.",
    "Should we move the standup?",
]
FACT_SHAPED = [
    "What database do we run?",
    "When did I join Reakon?",
    "What was the advice the lawyer gave me in March?",
    "How many invoices went out in May?",
]


@pytest.mark.parametrize("question", ADVICE_SHAPED)
async def test_an_advice_request_gets_the_advice_contract(question: str) -> None:
    seen: dict[str, str] = {}
    await answer(question, [_hit("They dislike loud rooms")], _capturing(seen))
    assert "no stored preference applies here" in seen["prompt"]
    assert "I don't have that in memory" not in seen["prompt"]


@pytest.mark.parametrize("question", FACT_SHAPED)
async def test_every_other_question_keeps_the_decline_contract(question: str) -> None:
    """The baseline this change must not touch. A fact question that lost rule 1
    would answer from the model's priors in the product's voice."""
    seen: dict[str, str] = {}
    await answer(question, [_hit("Postgres 16 on Cloud SQL")], _capturing(seen))
    assert "I don't have that in memory" in seen["prompt"]
    assert "no stored preference applies here" not in seen["prompt"]


#: Each of these was bought with a measurement, and a prompt is the easiest
#: thing in a codebase to "tidy" back to something shorter. Pinned by the
#: behaviour it produces, not by exact wording, so rephrasing stays allowed and
#: DELETING the mechanism does not.
@pytest.mark.parametrize(
    ("element", "why"),
    [
        ("EVERY", "exhaustiveness: 'the preference(s)' invites the model to find one"),
        ("AVOID", "a prohibition must rank with a taste, not below it"),
        ("CHECK", "naming a constraint does not stop the model violating it"),
        ("ALL of them at once", "the recommendation must satisfy the whole set"),
        ("latest", "an updated preference must not be applied in its old form"),
    ],
)
def test_the_advice_prompt_keeps_what_was_measured(element: str, why: str) -> None:
    assert element in ADVICE_CHAT_PROMPT, why


def test_the_two_contracts_are_not_accidentally_the_same() -> None:
    """A copy-paste that left both templates identical would make every test
    above pass while the routing did nothing."""
    assert ADVICE_CHAT_PROMPT != CHAT_PROMPT
    assert "Answer ONLY from the memories" in CHAT_PROMPT
    assert "Answer ONLY from the memories" not in ADVICE_CHAT_PROMPT


async def test_history_reaches_the_prompt() -> None:
    seen: dict[str, str] = {}
    await answer(
        "and the cache?",
        [_hit("Redis 7 backs the rate limiter")],
        _capturing(seen),
        history=[Turn(role="user", content="what database?")],
    )
    assert "Conversation so far:" in seen["prompt"]
    assert "what database?" in seen["prompt"]


async def test_considered_lists_every_memory_that_reached_the_prompt() -> None:
    """Cited or not, so a caller can judge the ones the model ignored."""
    hits = [_hit("a"), _hit("b"), _hit("c")]
    result = await answer("q", hits, _replying("only [2] matters"))
    assert result.considered == [h.memory.id for h in hits]
    assert result.cited == [hits[1].memory.id]


async def test_using_an_unverified_memory_is_flagged() -> None:
    """The caveat has to travel with the answer, not just into the prompt."""
    shaky = _hit("revenue was 4.2M", metadata={"confidence": "unverified"})
    solid = _hit("headcount is 12")
    result = await answer("revenue?", [shaky, solid], _replying("4.2M per [1]"))
    assert result.used_unverified is True


async def test_citing_only_verified_memories_is_not_flagged() -> None:
    """The flag tracks what was USED, not what was available -- otherwise one
    shaky memory in the window would taint every answer from that space."""
    shaky = _hit("revenue was 4.2M", metadata={"confidence": "unverified"})
    solid = _hit("headcount is 12")
    result = await answer("headcount?", [shaky, solid], _replying("12 per [2]"))
    assert result.used_unverified is False


async def test_an_uncited_answer_over_an_unverified_memory_is_not_flagged() -> None:
    """No citation means nothing was used, so nothing unverified was used."""
    shaky = _hit("revenue was 4.2M", metadata={"confidence": "unverified"})
    result = await answer("revenue?", [shaky], _replying("I don't have that in memory."))
    assert result.used_unverified is False
    assert result.cited == []


async def test_the_reply_is_stripped() -> None:
    result = await answer("q", [_hit("a")], _replying("  padded [1]  \n"))
    assert result.reply == "padded [1]"


async def test_an_empty_completion_yields_an_empty_reply_not_a_crash() -> None:
    result = await answer("q", [_hit("a")], _replying(""))
    assert result.reply == ""
    assert result.cited == []


async def test_a_completion_failure_propagates() -> None:
    """Deliberately NOT fail-open, unlike derive and expansion.

    Those degrade to a worse answer. There is no degraded chat -- returning a
    cheerful reply after the model failed would be inventing one, which is the
    single thing this module exists to prevent. The caller gets the error and
    can decide.
    """

    async def broken(prompt: str) -> str:
        raise RuntimeError("vendor is down")

    with pytest.raises(RuntimeError, match="vendor is down"):
        await answer("q", [_hit("a")], broken)


async def test_a_hundred_memories_are_all_citable_end_to_end() -> None:
    """The regression the coverage fix would otherwise have introduced."""
    hits = [_hit(f"fact number {i}") for i in range(150)]
    result = await answer("q", hits, _replying("see [1], [50] and [150]"))
    assert result.cited == [hits[0].memory.id, hits[49].memory.id, hits[149].memory.id]


def test_the_prompt_template_has_the_three_placeholders() -> None:
    """A missing placeholder is a KeyError at request time, not at import."""
    for field in ("{history}", "{memories}", "{question}"):
        assert field in CHAT_PROMPT


def test_the_answer_dataclass_defaults_are_safe_to_read() -> None:
    """A caller reading `.cited` on a default-constructed answer must not get
    None -- an `if not answer.cited` branch would behave the same, but a
    `for id in answer.cited` would raise."""
    blank = ChatAnswer(reply="x")
    assert blank.cited == []
    assert blank.considered == []
    assert blank.used_unverified is False


# -- advice routing: the measured prompt swap ------------------------------
#
# The preference lab measured the decline rule as the advice failure mode:
# fact framing 7/12 with three refusals, advice framing 10/12, identical
# evidence. `answer()` now routes by question shape -- same classifier as the
# derive path, no caller flag -- so these pin which contract each shape gets.


async def test_an_advice_request_gets_the_grounded_advice_contract() -> None:
    seen: dict[str, str] = {}
    await answer(
        "Can you recommend a hotel for my Barcelona trip?",
        [_hit("The Lisbon hotel's rooftop pool sold me instantly.")],
        _capturing(seen),
    )
    prompt = seen["prompt"]
    assert "advising this user" in prompt
    assert "no stored preference applies" in prompt
    assert "I don't have that in memory" not in prompt, (
        "the decline instruction reached an advice request -- the measured "
        "failure mode (three refusals of twelve) is back"
    )


async def test_a_fact_question_keeps_the_decline_contract() -> None:
    """The routing must not soften the fact path: inventing a fact is worse
    than refusing one, and the decline rule is correct there."""
    seen: dict[str, str] = {}
    await answer(
        "What database do we use?",
        [_hit("Postgres 16 on Cloud SQL.")],
        _capturing(seen),
    )
    assert "I don't have that in memory" in seen["prompt"]
    assert "advising this user" not in seen["prompt"]


async def test_advice_citations_still_parse_and_flag_unverified() -> None:
    """The two-step advice shape keeps the [n] convention, so the citation
    machinery -- and the unverified flag riding on it -- work unchanged."""
    shaky = _hit("I think I preferred the aisle?", metadata={"confidence": "unsure"})
    result = await answer(
        "Which seat should I book for the long flight?",
        [shaky],
        _replying('Per [1] "preferred the aisle" -- book the aisle.'),
    )
    assert result.cited == [shaky.memory.id]
    assert result.used_unverified is True


async def test_an_advice_request_over_an_empty_store_still_declines() -> None:
    """ "Never refuse" applies to thin evidence, not to NO evidence: with an
    empty store there are no preferences to apply, and a recommendation from
    nothing is the model's own priors wearing the product's voice."""
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "I recommend the aisle!"

    result = await answer("Can you recommend a seat for me?", [], complete)
    assert not called
    assert "don't have" in result.reply
