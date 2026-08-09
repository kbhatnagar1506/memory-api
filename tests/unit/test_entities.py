"""Entity extraction for cross-session bridging."""

from __future__ import annotations

from mapi.domain.retrieval.entities import (
    coverage,
    extract_entities,
    salient_entities,
)


def test_finds_proper_nouns() -> None:
    found = extract_entities("I met Caroline at State University last year")
    assert "caroline" in found
    assert any("state university" in f for f in found)


def test_sentence_initial_name_needs_confirmation_elsewhere() -> None:
    """Alone, a sentence-opening name is shape-identical to "Wow"."""
    assert extract_entities("Caroline enrolled at State") == ["state"]
    # Pooled across seeds, one mid-sentence use confirms it everywhere.
    pooled = salient_entities(["Caroline enrolled at State", "I spoke to Caroline about it"])
    assert "caroline" in pooled


def test_strips_speaker_prefix() -> None:
    """A speaker prefix names the speaker, not a topic. In a two-person
    dialogue every turn has one, so counting it makes everything match."""
    assert "caroline" not in extract_entities("Caroline: I went to the shop")


def test_ignores_sentence_initial_grammar_words() -> None:
    for text in ("The meeting was long", "There was a delay", "Monday was busy"):
        assert extract_entities(text) == [], text


def test_finds_acronyms_and_codes() -> None:
    found = extract_entities("We hit error TS2345 in the NASA integration")
    assert "ts2345" in found
    assert "nasa" in found


def test_finds_years() -> None:
    assert "2023" in extract_entities("She graduated in 2023")


def test_ignores_small_numbers() -> None:
    assert "42" not in extract_entities("I have 42 apples")


def test_empty_and_blank() -> None:
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_ranked_by_frequency() -> None:
    text = "Melanie called Melanie about Melanie. Also Dave."
    found = extract_entities(text)
    assert found[0] == "melanie"


def test_respects_max_entities() -> None:
    text = " ".join(f"Person{i}name" for i in range(30))
    assert len(extract_entities(text, max_entities=5)) <= 5


# -- salient_entities ----------------------------------------------------------


def test_drops_entities_already_in_the_query() -> None:
    """Expanding on a query term re-fetches the neighbourhood we already have."""
    seeds = ["Caroline studied psychology at Stanford"]
    out = salient_entities(seeds, query="What did Caroline study?")
    assert "caroline" not in out
    assert any("stanford" in e or "psychology" in e for e in out) or out == []


def test_drops_ubiquitous_entities() -> None:
    """Both speakers appear in every turn of a dialogue; expanding on them
    returns the whole corpus."""
    seeds = [f"Melanie mentioned topic {i} to Melanie" for i in range(10)]
    assert "melanie" not in salient_entities(seeds)


def test_keeps_entities_that_appear_in_a_minority_of_seeds() -> None:
    seeds = [
        "Talked about Stanford yesterday",
        "The weather was fine",
        "Nothing much happened",
        "Quiet day overall",
    ]
    assert "stanford" in salient_entities(seeds)


def test_empty_seeds() -> None:
    assert salient_entities([]) == []


def test_respects_limit() -> None:
    seeds = [f"Entity{i}name appeared once" for i in range(20)]
    assert len(salient_entities(seeds, max_entities=3)) <= 3


def test_coverage() -> None:
    assert coverage(["caroline"], "Caroline went home") == 1.0
    assert coverage(["caroline", "dave"], "Caroline went home") == 0.5
    assert coverage([], "anything") == 0.0
