"""Ordering: the reducer always returned the LAST row, including for "first".

Found by reading the 89 failures of a 494-question run rather than by testing.
ORDER-shaped questions were wrong 16 times out of 62 -- the worst rate of any
non-advice shape, and worse than date arithmetic (12/102), which is where the
previous commit spent its effort.

Nine of those 16 were binary "which came first, A or B", and every single one
named the wrong one of the two:

    Which show did I start watching first, 'The Crown' or 'Game of Thrones'?
        gold Game of Thrones        said The Crown
    Which group did I join first, 'Page Turners' or 'Marketing Professionals'?
        gold Page Turners           said Marketing Professionals
    Which gift did I buy first, the necklace or the photo album?
        gold the photo album        said the necklace

A coin flip gets four or five of nine. Nine of nine inverted is a systematic
error, and the cause was one line: the reducer sorted the table and returned
`dated[-1]`, under a comment claiming "the question's own wording picks the
end". The code never read the wording -- it could not, because `question` was
not a parameter of the reducer at all until date arithmetic needed one.

The tests below are that defect, the sequence case it also could not express,
and the recency default it must not lose.
"""

from __future__ import annotations

from datetime import date

import pytest

from mapi.domain.synthesis.classify import QuestionKind
from mapi.domain.synthesis.derive import Extraction, _order_direction, _reduce_in_code

ASKED = date(2023, 6, 1)


def _row(when: date, fact: str) -> Extraction:
    return Extraction(date=when, fact=fact, quote=fact, source_id="doc")


#: Deliberately given to the reducer NEWEST FIRST, so a test that passes cannot
#: be passing because the input happened to be in the right order already.
PAIR = [
    _row(date(2023, 3, 4), "joined Marketing Professionals"),
    _row(date(2023, 1, 10), "joined Page Turners"),
]


# -- direction, read off the question --------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        # The nine real inversions, condensed to their asking shape.
        "Which show did I start watching first, 'The Crown' or 'Game of Thrones'?",
        "Which group did I join first, 'Page Turners' or 'Marketing Professionals'?",
        "Which event happened first, losing the charger or receiving the case?",
        "Who did I meet first, the woman selling jam or the tourist from Australia?",
        "Which gift did I buy first, the necklace or the photo album?",
        "Which project did I start first, the Ferrari model or the Zero model?",
        "Which invoice was dated earliest in January?",
        "What was the oldest thing in the archive?",
        "Which did I originally choose?",
    ],
)
def test_a_question_asking_which_came_first_wants_the_earliest(question: str) -> None:
    assert _order_direction(question).value == "earliest"


@pytest.mark.parametrize(
    "question",
    [
        "What was the last thing I submitted to a venue?",
        "Which streaming service did I start using most recently?",
        "What did I do most recently?",
        "What is the latest version?",
        "Which one did I buy newest?",
    ],
)
def test_a_question_asking_for_the_latest_still_gets_it(question: str) -> None:
    """The old behaviour was right for these and must not regress."""
    assert _order_direction(question).value == "latest"


@pytest.mark.parametrize(
    "question",
    [
        "What is the order of the six museums I visited from earliest to latest?",
        "What is the order of the three trips I took, from earliest to latest?",
        "In what order did I visit them?",
        "List them in chronological order.",
        "What was the sequence of events?",
    ],
)
def test_a_request_for_the_whole_ordering_is_neither_endpoint(question: str) -> None:
    """ "Earliest to latest" names both ends and is asking for neither.

    This shape could not be expressed at all before: it returned one row, so
    "the order of the six museums" was answered with one museum.
    """
    assert _order_direction(question).value == "sequence"


def test_a_question_naming_no_direction_defaults_to_the_most_recent() -> None:
    """Unchanged from before, so nothing that used to work stops working."""
    assert _order_direction("Which service did I use?").value == "latest"


# -- through the reducer ----------------------------------------------------


def test_the_real_inverted_case_now_answers_the_gold_value() -> None:
    got = _reduce_in_code(
        QuestionKind.ORDER,
        list(PAIR),
        "Which group did I join first, 'Page Turners' or 'Marketing Professionals'?",
        ASKED,
    )
    assert got is not None
    assert "Page Turners" in got
    assert "Marketing" not in got


def test_the_same_table_answers_the_opposite_question_oppositely() -> None:
    """The table is identical; only the question changed.

    This is the property the old code could not have: one input, two questions,
    two different correct answers.
    """
    got = _reduce_in_code(
        QuestionKind.ORDER, list(PAIR), "Which group did I join most recently?", ASKED
    )
    assert got is not None
    assert "Marketing" in got
    assert "Page Turners" not in got


def test_a_sequence_question_returns_every_row_in_date_order() -> None:
    got = _reduce_in_code(
        QuestionKind.ORDER,
        list(PAIR),
        "What is the order of the groups I joined from earliest to latest?",
        ASKED,
    )
    assert got is not None
    assert got.index("Page Turners") < got.index("Marketing")


def test_an_undated_table_declines_rather_than_ordering_arbitrarily() -> None:
    rows = [Extraction(date=None, fact="joined something", quote="q", source_id="d")]
    assert _reduce_in_code(QuestionKind.ORDER, rows, "Which came first?", ASKED) is None
