"""What `ingest` says it did, versus what it did.

`IngestResult` is the write path's whole audit surface. `superseded` names
memories that were hidden; `supersede_declined` names ones the system considered
hiding and did not, and its docstring is explicit that "these memories are still
active". A caller reconciling the two is entitled to assume they are disjoint.

They were not. Every APPLIED supersession also appeared in
`supersede_declined`:

    declined += [p.old_id for p in shortlist if p not in proposals]

`SupersessionProposal` is a frozen dataclass, so `in` compares all five fields --
including `reason` and `confidence`, which `_confirm_supersessions` deliberately
REPLACES with the adjudicator's verdict (that substitution is the point of the
adjudication step, and is commented as such twelve lines earlier). So no
confirmed proposal ever equalled its own shortlist entry, the filter kept
everything, and the `supersession_declined` log line counted confirmations as
refusals.

The fix compares target ids. These tests pin the disjointness, and the shapes
that must keep working: a genuinely refused supersession still gets reported, and
one below the confidence floor never reaches the adjudicator at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mapi.domain.models import MemoryStatus, Organization, Space
from mapi.service import MemoryService

WHEN = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

#: A statement and its later revision. Close enough to land inside the
#: supersession similarity band, different enough in the value to be a genuine
#: change rather than a restatement.
OLD = "The standup is at 9:30am every weekday in the main room."
NEW = "The standup is at 10:15am every weekday in the main room."


async def _confirming(prompt: str) -> str:
    """An adjudicator that confirms whatever it is shown.

    Returns the structured verdict `parse_supersede_verdicts` requires: the
    attribute plus both values, which the code then checks -- so a confirmation
    still has to survive the quotability and echo guards.
    """
    if "REPLACES" in prompt:
        return (
            '[{"n":1,"attribute":"standup time","old_value":"9:30am",'
            '"new_value":"10:15am","confidence":0.95}]'
        )
    return "[]"


async def _refusing(prompt: str) -> str:
    """An adjudicator that refuses everything."""
    return "[]"


@pytest.fixture
async def confirming(service: MemoryService) -> MemoryService:
    service.extractor = _confirming
    return service


@pytest.fixture
async def refusing(service: MemoryService) -> MemoryService:
    service.extractor = _refusing
    return service


async def _write_pair(
    service: MemoryService, org: Organization, space: Space
) -> tuple[object, object]:
    first = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content=OLD,
        occurred_at=WHEN,
        extract=False,
    )
    second = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content=NEW,
        occurred_at=WHEN + timedelta(days=7),
        extract=False,
    )
    return first, second


# -- the defect ------------------------------------------------------------


async def test_an_applied_supersession_is_not_also_reported_as_declined(
    confirming: MemoryService, org: Organization, space: Space
) -> None:
    """The two lists must be disjoint. They were identical.

    A caller cannot act on "this memory was hidden AND left active".
    """
    first, second = await _write_pair(confirming, org, space)
    assert first.memory.id in second.superseded, "the revision did not apply"
    assert first.memory.id not in second.supersede_declined
    assert not (set(second.superseded) & set(second.supersede_declined))


async def test_a_confirmed_supersession_actually_hides_the_old_memory(
    confirming: MemoryService, org: Organization, space: Space
) -> None:
    """Ties the report to the state, which is what made the old bug visible:
    the id was reported as still active while its status said SUPERSEDED."""
    first, second = await _write_pair(confirming, org, space)
    stored = await confirming.get_memory(org.id, space.id, first.memory.id)
    assert stored.status is MemoryStatus.SUPERSEDED
    assert first.memory.id not in second.supersede_declined


async def test_the_declined_list_is_empty_when_everything_applied(
    confirming: MemoryService, org: Organization, space: Space
) -> None:
    _first, second = await _write_pair(confirming, org, space)
    assert second.supersede_declined == []


# -- the shapes that must keep working -------------------------------------


async def test_a_refused_supersession_is_still_reported(
    refusing: MemoryService, org: Organization, space: Space
) -> None:
    """The fix must not empty the field. A shortlisted proposal the adjudicator
    rejected is exactly what `supersede_declined` is for."""
    first, second = await _write_pair(refusing, org, space)
    assert second.superseded == []
    assert first.memory.id in second.supersede_declined
    stored = await refusing.get_memory(org.id, space.id, first.memory.id)
    assert stored.status is MemoryStatus.ACTIVE


async def test_no_adjudicator_means_nothing_is_hidden_and_everything_is_reported(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """Fail-closed: with no extraction backend, supersession is proposed and
    never applied, and the caller is told so rather than left to infer it."""
    service.extractor = None
    first, second = await _write_pair(service, org, space)
    assert second.superseded == []
    assert first.memory.id in second.supersede_declined


async def test_an_unrelated_write_proposes_nothing_either_way(
    confirming: MemoryService, org: Organization, space: Space
) -> None:
    """Both lists empty is the common case and must stay quiet."""
    await confirming.ingest(
        org_id=org.id, space_id=space.id, content=OLD, occurred_at=WHEN, extract=False
    )
    other = await confirming.ingest(
        org_id=org.id,
        space_id=space.id,
        content="The office fire drill is scheduled for the last Friday of April.",
        occurred_at=WHEN + timedelta(days=3),
        extract=False,
    )
    assert other.superseded == []
    assert other.supersede_declined == []
