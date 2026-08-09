"""Text analysis.

These pin the semantics both storage backends must share. PostgreSQL's `english`
configuration stems and removes stopwords; the in-memory BM25 index has to do
the same or the two disagree about what matches.
"""

from __future__ import annotations

import pytest

from mapi.domain.text import STOPWORDS, analyze, stem, tokenize


@pytest.mark.parametrize(
    "word,expected",
    [
        ("deploys", "deploy"),
        ("deployed", "deploy"),
        ("deploying", "deploy"),
        ("cats", "cat"),
        ("flies", "fly"),
        ("running", "run"),
        ("stopped", "stop"),
        ("quickly", "quick"),
        ("upgrades", "upgrade"),
    ],
)
def test_inflections_collapse_to_one_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


@pytest.mark.parametrize("word", ["is", "be", "the", "bus", "gas", "yes"])
def test_short_words_are_left_alone(word: str) -> None:
    """Over-eager stemming turns 'bus' into 'bu' and breaks exact matching."""
    assert stem(word) == word


def test_query_and_document_share_a_stem() -> None:
    """The bug this module exists to fix: 'deploy' must match 'Deploys'."""
    query = set(analyze("how do we deploy?"))
    document = set(analyze("Deploys go through Jenkins"))
    assert query & document == {"deploy"}


def test_stopwords_are_removed_from_documents() -> None:
    """Otherwise 'how do we deploy' matches any document containing 'we'."""
    assert "we" not in analyze("We chose Kafka for the event bus")
    assert "the" not in analyze("the quick brown fox")


def test_all_stopword_text_analyzes_to_nothing() -> None:
    """An empty result is a real signal, not something to paper over."""
    assert analyze("how do we") == []


def test_query_analysis_falls_back_rather_than_matching_nothing() -> None:
    from mapi.domain.text import analyze_query

    assert analyze_query("how do we") == ["how", "do", "we"]
    assert analyze_query("deploy") == ["deploy"]


def test_tokenize_keeps_stopwords() -> None:
    assert "the" in tokenize("the cat")


def test_analysis_is_case_insensitive() -> None:
    assert analyze("KAFKA Cluster") == analyze("kafka cluster")


def test_unicode_is_tokenized() -> None:
    assert analyze("café naïve") == ["café", "naïve"]


def test_empty_input() -> None:
    assert analyze("") == []
    assert tokenize("") == []


def test_punctuation_does_not_create_junk_tokens() -> None:
    """ "what's" must analyze to "what", never to the unmatchable "what'"."""
    assert tokenize("what's the error?") == ["what", "the", "error"]
    assert analyze("error! the error?") == ["error", "error"]
    assert all("'" not in t for t in analyze("don't 'quote' me"))


def test_stopword_list_is_conservative() -> None:
    """Words that carry meaning in a technical corpus must survive."""
    for word in ("can", "bus", "data", "log", "run", "error", "api"):
        assert word not in STOPWORDS


def test_stem_is_idempotent() -> None:
    for word in ("deploys", "running", "quickly", "cats"):
        assert stem(stem(word)) == stem(word)
