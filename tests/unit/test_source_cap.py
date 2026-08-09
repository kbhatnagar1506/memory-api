"""Per-source capping: coverage over raw score in the top-k.

Write-time extraction turns one document into ~13 retrievable units. Ranked
purely by score, two or three documents then eat the whole window -- measured
on LongMemEval as full_recall@k 0.968 -> 0.948 while MRR ROSE to 0.987, and
21 questions lost end to end, 17 of them in the two capabilities that need
several distinct sessions at once.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mapi.domain.models import Memory, ScoredMemory
from mapi.domain.retrieval.pipeline import _cap_per_source, source_of

WHEN = datetime(2023, 5, 1, tzinfo=UTC)

#: Memory ids are validated prefixed ULIDs, so a test cannot invent "a1".
#: Labels map onto generated ids and back, keeping assertions readable.
_LABELS: dict[str, str] = {}


def _m(label: str, source: str | None, score: float) -> ScoredMemory:
    memory = Memory(
        org_id="org_1",
        space_id="spc_1",
        content=label,
        metadata={"extracted_from": source} if source else {},
        occurred_at=WHEN,
    )
    _LABELS[memory.id] = label
    return ScoredMemory(memory=memory, score=score)


def labels(results: list[ScoredMemory]) -> list[str]:
    return [_LABELS[s.memory.id] for s in results]


def test_a_dominant_source_cannot_eat_the_window() -> None:
    scored = [_m(f"a{i}", "doc_a", 1.0 - i / 100) for i in range(8)]
    scored += [_m("b1", "doc_b", 0.5), _m("c1", "doc_c", 0.4)]

    kept = _cap_per_source(scored, 2)

    assert len({source_of(s.memory) for s in kept}) == 3
    assert sum(1 for s in kept if source_of(s.memory) == "doc_a") == 2


def test_each_source_keeps_its_best_units() -> None:
    """Capping must drop a source's WORST units, never its best."""
    scored = [_m("hi", "doc_a", 0.9), _m("mid", "doc_a", 0.5), _m("lo", "doc_a", 0.1)]
    assert labels(_cap_per_source(scored, 2)) == ["hi", "mid"]


def test_relative_order_is_preserved() -> None:
    scored = [_m("a1", "doc_a", 0.9), _m("b1", "doc_b", 0.8), _m("a2", "doc_a", 0.7)]
    assert labels(_cap_per_source(scored, 1)) == ["a1", "b1"]


def test_undecomposed_memories_are_never_grouped() -> None:
    """A corpus with no extraction must be unaffected: each memory is its own
    source, so no cap can ever bind."""
    scored = [_m(f"m{i}", None, 1.0 - i / 10) for i in range(6)]
    assert len(_cap_per_source(scored, 1)) == 6


def test_benchmark_doc_id_is_recognised_as_a_source() -> None:
    """The harness records provenance as `doc_id`, not `extracted_from`."""
    memory = Memory(
        org_id="org_1",
        space_id="spc_1",
        content="c",
        metadata={"doc_id": "session_7"},
        occurred_at=WHEN,
    )
    assert source_of(memory) == "session_7"


def test_explain_records_why_a_result_survived() -> None:
    scored = [_m("a1", "doc_a", 0.9), _m("a2", "doc_a", 0.8)]
    kept = _cap_per_source(scored, 2)
    assert any("contributed 2/2" in note for note in kept[1].explain)
