"""Per-tenant limits on what a write may consume.

Rate limiting already caps how FAST a tenant can call. It says nothing about
how much they can accumulate, and the two failures are different: a rate
limit protects the service from a burst, a quota protects it from a tenant
who is simply large. Without one, a single organization can grow a space
without bound, and every cost that scales with corpus size -- storage,
embedding calls, the candidate scan on every write -- scales with it.

Three limits, because three different things run out:

  * MEMORIES per organization. The retrieval cost driver: an unbounded
    corpus is an unbounded ANN index, and recall degrades long before storage
    does. Counted per ORG rather than per space, because a per-space limit is
    escaped by creating another space.
  * BYTES per organization. The storage cost driver, counted on normalized
    content so it matches what is actually persisted.
  * WRITES per day per organization. The vendor-spend driver: every write
    that is not a duplicate is at least one embedding call, and with
    extraction on it is one completion too. This is the limit that stands
    between one enthusiastic customer and a surprising Vertex invoice.

All three default to OFF (0 = unlimited). A memory API that shipped with
opinions about how much its users may remember would be wrong for almost
everyone, and the operator setting a number has context this code does not.

Counting is deliberately approximate at the edges. `count_memories` is a
point-in-time read, so two concurrent writes can both observe
`count == limit - 1` and both proceed. The overshoot is bounded by
concurrency, and the alternative -- serializing every write behind a lock or
a SELECT FOR UPDATE on a counter row -- costs more than the last memory over
the line is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MapiError


class QuotaExceededError(MapiError):
    """A tenant limit, not a rate limit.

    402 rather than 429: the request was well-formed and the caller was not
    going too fast. Retrying will not help, which is exactly what 429 tells a
    client to do -- and a client that retries a quota failure on a backoff
    loop will hammer the endpoint forever. The remedy is a bigger plan or
    fewer memories, and the payload names which limit was hit so the caller
    can tell those apart without parsing prose.
    """

    status_code = 402
    slug = "quota_exceeded"
    title = "Quota exceeded"

    def __init__(self, detail: str, *, limit_name: str, limit: int, current: int) -> None:
        super().__init__(
            detail,
            extra={"limit_name": limit_name, "limit": limit, "current": current},
        )
        self.limit_name = limit_name


@dataclass(frozen=True, slots=True)
class Quota:
    """The limits in force for one tenant. 0 means unlimited."""

    max_memories_per_org: int = 0
    max_bytes_per_org: int = 0
    max_writes_per_day: int = 0

    @property
    def enforced(self) -> bool:
        """Whether any limit is set at all.

        Checked before the counting queries run: with everything unlimited,
        enforcement must cost zero, or every deployment pays for a feature
        most of them have not turned on.
        """
        return bool(
            self.max_memories_per_org or self.max_bytes_per_org or self.max_writes_per_day
        )


def check_memories(quota: Quota, current: int) -> None:
    if quota.max_memories_per_org and current >= quota.max_memories_per_org:
        raise QuotaExceededError(
            f"this organization holds {current} memories, and the limit is "
            f"{quota.max_memories_per_org}",
            limit_name="max_memories_per_org",
            limit=quota.max_memories_per_org,
            current=current,
        )


def check_bytes(quota: Quota, current: int, incoming: int) -> None:
    if quota.max_bytes_per_org and current + incoming > quota.max_bytes_per_org:
        raise QuotaExceededError(
            f"this write would take the organization to {current + incoming} bytes, "
            f"and the limit is {quota.max_bytes_per_org}",
            limit_name="max_bytes_per_org",
            limit=quota.max_bytes_per_org,
            current=current,
        )


def check_writes(quota: Quota, current: int) -> None:
    if quota.max_writes_per_day and current >= quota.max_writes_per_day:
        raise QuotaExceededError(
            f"this organization has made {current} writes today, and the limit is "
            f"{quota.max_writes_per_day}",
            limit_name="max_writes_per_day",
            limit=quota.max_writes_per_day,
            current=current,
        )


__all__ = [
    "Quota",
    "QuotaExceededError",
    "check_bytes",
    "check_memories",
    "check_writes",
]
