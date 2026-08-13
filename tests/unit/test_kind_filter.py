"""Filtering retrieval by memory kind.

The point of the whole filter: write-time extraction stores claims beside
episodes, and four LongMemEval arms agree the cost lands when claims are
RETRIEVED rather than stored. `only` retrieved BETTER than `add` (full_recall
0.950 vs 0.948) and answered WORSE (0.762 vs 0.781) -- the difference is
entirely what the answerer reads.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mapi.domain.models import Memory, MemoryKind
from mapi.store.base import MemoryFilter

WHEN = datetime(2023, 5, 1, tzinfo=UTC)


def _m(kind: MemoryKind) -> Memory:
    return Memory(org_id="org_1", space_id="spc_1", content="x", kind=kind, occurred_at=WHEN)


def test_no_kinds_means_every_kind() -> None:
    """The default must not change behaviour for anyone."""
    f = MemoryFilter()
    assert f.matches(_m(MemoryKind.EPISODIC))
    assert f.matches(_m(MemoryKind.DERIVED))


def test_episodes_only_excludes_claims() -> None:
    f = MemoryFilter(kinds=frozenset({MemoryKind.EPISODIC}))
    assert f.matches(_m(MemoryKind.EPISODIC))
    assert not f.matches(_m(MemoryKind.DERIVED))


def test_claims_only_excludes_episodes() -> None:
    f = MemoryFilter(kinds=frozenset({MemoryKind.DERIVED}))
    assert f.matches(_m(MemoryKind.DERIVED))
    assert not f.matches(_m(MemoryKind.EPISODIC))


def test_kind_filtering_composes_with_status() -> None:
    """Filters are conjunctive; a kind match must not rescue a wrong status."""
    from mapi.domain.models import MemoryStatus

    f = MemoryFilter(
        kinds=frozenset({MemoryKind.DERIVED}),
        statuses=frozenset({MemoryStatus.ACTIVE}),
    )
    stale = _m(MemoryKind.DERIVED).model_copy(update={"status": MemoryStatus.STALE})
    assert not f.matches(stale)
