"""Deduplication and belief revision."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from supermemory.domain.consolidation import (
    DuplicateKind,
    apply_supersession,
    detect_exact_duplicate,
    detect_near_duplicate,
    merge_duplicate,
    propose_supersessions,
)
from supermemory.domain.models import Memory, MemoryStatus, RelationType

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make(content: str, *, days_ago: int = 0, **kw) -> Memory:
    return Memory(
        org_id="org_00000000000000000000000000",
        space_id="spc_00000000000000000000000000",
        content=content,
        occurred_at=NOW - timedelta(days=days_ago),
        **kw,
    )


def test_exact_duplicate_ignores_whitespace_and_case() -> None:
    stored = make("The Cat   Sat")
    verdict = detect_exact_duplicate("the cat sat", [stored])
    assert verdict.kind is DuplicateKind.EXACT
    assert verdict.existing_id == stored.id


def test_exact_duplicate_skips_archived() -> None:
    stored = make("hello", status=MemoryStatus.ARCHIVED)
    assert not detect_exact_duplicate("hello", [stored]).is_duplicate


def test_exact_duplicate_absent() -> None:
    assert not detect_exact_duplicate("hello", [make("goodbye")]).is_duplicate


def test_near_duplicate_above_threshold() -> None:
    verdict = detect_near_duplicate([1.0, 0.0], [("m1", [0.999, 0.001])], threshold=0.97)
    assert verdict.kind is DuplicateKind.NEAR
    assert verdict.existing_id == "m1"


def test_near_duplicate_below_threshold_reports_best_similarity() -> None:
    verdict = detect_near_duplicate([1.0, 0.0], [("m1", [0.0, 1.0])], threshold=0.97)
    assert not verdict.is_duplicate
    assert verdict.similarity >= 0.0


def test_near_duplicate_skips_dimension_mismatches() -> None:
    """A stale index built by another model must not crash a write."""
    verdict = detect_near_duplicate([1.0, 0.0], [("m1", [1.0, 0.0, 0.0])])
    assert not verdict.is_duplicate


def test_near_duplicate_empty_inputs() -> None:
    assert not detect_near_duplicate([], [("m", [1.0])]).is_duplicate
    assert not detect_near_duplicate([1.0], []).is_duplicate


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_near_duplicate_rejects_bad_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        detect_near_duplicate([1.0], [("m", [1.0])], threshold=threshold)


def test_supersession_requires_the_new_memory_to_be_newer() -> None:
    older = make("Deploys use Jenkins", days_ago=1)
    newer = make("Deploys now use GitHub Actions instead", days_ago=100)
    # `newer` is actually older here, so nothing may be proposed.
    assert propose_supersessions(newer, [1.0, 0.0], [(older, [0.9, 0.1])]) == []


def test_supersession_proposed_in_the_similarity_band() -> None:
    old = make("Deploys go through Jenkins", days_ago=100)
    new = make("Deploys now go through GitHub Actions instead of Jenkins")
    proposals = propose_supersessions(new, [1.0, 0.0], [(old, [0.85, 0.53])])
    assert proposals
    assert proposals[0].old_id == old.id
    assert 0.0 < proposals[0].confidence <= 0.99


def test_revision_language_raises_confidence() -> None:
    old = make("Deploys go through Jenkins", days_ago=100)
    plain = make("Deploys go through GitHub Actions")
    marked = make("Deploys no longer go through Jenkins; replaced by GitHub Actions")
    vector, old_vector = [1.0, 0.0], [0.85, 0.53]
    a = propose_supersessions(plain, vector, [(old, old_vector)])
    b = propose_supersessions(marked, vector, [(old, old_vector)])
    assert b[0].confidence > a[0].confidence


def test_near_identical_memories_are_duplicates_not_supersessions() -> None:
    """Above the high band they are restatements, which dedup handles."""
    old = make("Deploys go through Jenkins", days_ago=100)
    new = make("Deploys go through Jenkins")
    assert propose_supersessions(new, [1.0, 0.0], [(old, [1.0, 0.0])]) == []


def test_unrelated_memories_are_never_superseded() -> None:
    old = make("Team offsite in Lisbon", days_ago=100)
    new = make("Deploys go through GitHub Actions")
    assert propose_supersessions(new, [1.0, 0.0], [(old, [0.0, 1.0])]) == []


def test_supersession_skips_non_active_candidates() -> None:
    old = make("Old fact", days_ago=100, status=MemoryStatus.SUPERSEDED)
    new = make("New fact instead")
    assert propose_supersessions(new, [1.0, 0.0], [(old, [0.85, 0.53])]) == []


def test_apply_supersession_builds_an_edge_and_marks_the_old_memory() -> None:
    old = make("old", days_ago=10)
    new = make("new")
    proposals = propose_supersessions(new, [1.0, 0.0], [(old, [0.85, 0.53])])
    edge, updated_old = apply_supersession(new, old, proposals[0])

    assert updated_old.status is MemoryStatus.SUPERSEDED
    assert updated_old.version == old.version + 1
    # Direction matters: source is the NEWER memory.
    assert edge.type is RelationType.SUPERSEDES
    assert edge.source_id == new.id
    assert edge.target_id == old.id
    assert edge.confidence == proposals[0].confidence


def test_apply_supersession_does_not_bump_the_new_memory() -> None:
    """An edge is a fact about the graph, not an edit to the memory.

    Bumping the source's version here would fill its history with entries
    whose content is byte-identical.
    """
    old = make("old", days_ago=10)
    new = make("new")
    proposal = propose_supersessions(new, [1.0, 0.0], [(old, [0.85, 0.53])])[0]
    apply_supersession(new, old, proposal)
    assert new.version == 1


def test_apply_supersession_is_deterministic_in_its_edge() -> None:
    """Called twice, it describes the same relation both times.

    Deduplication itself is the store's job (create_relation is idempotent);
    what matters here is that the endpoints and type never vary.
    """
    old = make("old", days_ago=10)
    new = make("new")
    proposal = propose_supersessions(new, [1.0, 0.0], [(old, [0.85, 0.53])])[0]
    first, _ = apply_supersession(new, old, proposal)
    second, _ = apply_supersession(new, old, proposal)
    assert (first.source_id, first.target_id, first.type) == (
        second.source_id,
        second.target_id,
        second.type,
    )


def test_a_memory_cannot_supersede_itself() -> None:
    m = make("self")
    proposal = type("P", (), {"reason": "", "confidence": 0.5})()
    with pytest.raises(ValueError, match="itself"):
        apply_supersession(m, m, proposal)  # type: ignore[arg-type]


def test_merge_duplicate_unions_metadata_and_tags() -> None:
    existing = make("x", metadata={"a": 1}, tags=["one"])
    incoming = make("x", metadata={"b": 2}, tags=["two"])
    merged = merge_duplicate(existing, incoming)
    assert merged.metadata == {"a": 1, "b": 2}
    assert set(merged.tags) == {"one", "two"}
    assert merged.version == existing.version + 1


def test_merge_duplicate_keeps_the_newer_event_time() -> None:
    existing = make("x", days_ago=10)
    incoming = make("x", days_ago=1)
    assert merge_duplicate(existing, incoming).occurred_at == incoming.occurred_at
