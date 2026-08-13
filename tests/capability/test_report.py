"""The ledger: every headline capability number in one place, printed.

Run it and read it:

    pytest tests/capability -q -s -k report

The individual families assert per-case invariants. This asserts the AGGREGATE,
and it exists because a suite of a hundred green ticks does not tell you how good
retrieval is -- it tells you nothing broke. A number does.

Two rules make it survivable as a build gate:

**Bounds, not equalities.** `joint_recall@k >= 0.80`, never `== 0.9375`. A test
asserting an exact metric fails on every unrelated tuning change and is deleted
within a month, taking the coverage with it. The floors are set below the current
measurement with deliberate headroom.

**The whole table prints on failure.** A bound that trips tells you one number
moved; the table tells you which and by how much, which is the difference between
a useful failure and a bisect.

WHAT IS DELIBERATELY NOT HERE. No answer accuracy, no judge, no model. Those
belong to `bench/` and cost money. Everything below is computed from ranked ids
with a scripted embedder, so it runs in CI in milliseconds and cannot drift with a
vendor's weights.
"""

from __future__ import annotations

from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import (
    joint_recall_at_k,
    ndcg_at_k,
    recall_at_k,
    unique_information_at_k,
)
from tests.support.vectors import ANCHOR, at_cosine, axis, band, cone

from mapi.domain.retrieval.pipeline import SearchRequest

#: Floors, set below what the system measures today with headroom for noise. A
#: floor at the measured value would fail on any reordering; a floor far below it
#: would assert nothing. These sit roughly one bad case beneath current.
FLOORS: dict[str, float] = {
    "recall@10": 0.95,
    "joint_recall@10 (4 members)": 1.0,
    "joint_recall@10 (16 members)": 1.0,
    "ndcg@10": 0.90,
    "unique_information@5 under flooding": 2.0,
    "rank of gold among 64 hard negatives": 1.0,
}


async def _measure(geometry: tuple) -> dict[str, float]:
    """Every headline number, from one service over separate spaces."""
    service, org, _space, embedder = geometry
    out: dict[str, float] = {}

    async def run(slug: str, facts: list[Fact], gold: tuple[str, ...], limit: int) -> list[str]:
        """Load a corpus, search once, return the GOLD IDS in rank order."""
        space_n = await service.create_space(org.id, slug=slug, name=slug)
        corpus = build(
            facts,
            [
                Query(
                    key="q", text="the kelmady arrangement", vector=axis(ANCHOR), gold_ids=gold
                )
            ],
        )
        register(corpus, embedder)
        await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
        response = await service.search(
            SearchRequest(
                query="the kelmady arrangement",
                org_id=org.id,
                space_id=space_n.id,
                limit=limit,
                **GEOMETRY,
            )
        )
        return corpus.gold_of([hit.memory.id for hit in response.results])

    # recall@10 and ndcg@10: one gold among distractors.
    facts = [
        Fact(gold_id="gold", text="gold the kelmady arrangement", vector=at_cosine(0.90, off=1))
    ]
    facts += [
        Fact(gold_id=f"d{i}", text=f"d{i} filler", vector=v)
        for i, v in enumerate(band(0.60, 12, first_off=2))
    ]
    gold_order = await run("rep-recall", facts, ("gold",), 10)
    out["recall@10"] = recall_at_k(gold_order, ["gold"], 10)
    out["ndcg@10"] = ndcg_at_k(gold_order, ["gold"], 10)

    # joint_recall at two set sizes.
    for members in (4, 16):
        facts = [
            Fact(gold_id=f"g{i:02d}", text=f"g{i:02d} member {i}", vector=v)
            for i, v in enumerate(band(0.85, members, first_off=1))
        ]
        facts += [
            Fact(
                gold_id=f"f{i}",
                text=f"f{i} filler",
                vector=at_cosine(0.40, off=1 + members + i),
            )
            for i in range(6)
        ]
        required = tuple(f"g{i:02d}" for i in range(members))
        order = await run(f"rep-joint{members}", facts, required, 10 if members <= 4 else 20)
        out[f"joint_recall@10 ({members} members)"] = joint_recall_at_k(
            order, required, 10 if members <= 4 else 20
        )

    # unique_information@5 under a near-duplicate flood.
    #
    # The `fact_of` mapping is what makes this metric mean anything: all ten
    # duplicates map to ONE fact identity ("thraskin"), so ten copies in the
    # window count once. Mapping each row to itself -- which the first version of
    # this did -- reports 5 distinct facts for five rows and measures nothing,
    # since that is just `len`.
    facts = [
        Fact(
            gold_id="gold", text="gold the kelmady arrangement", vector=at_cosine(0.90, off=1)
        ),
        Fact(
            gold_id="other", text="other the thraskin schedule", vector=at_cosine(0.70, off=90)
        ),
    ]
    facts += [
        Fact(gold_id=f"dup{i}", text=f"dup{i} thraskin restated", vector=v)
        for i, v in enumerate(cone(0.88, 0.95, 10, first_off=2))
    ]
    order = await run("rep-flood", facts, ("gold",), 5)
    identity = {"gold": "kelmady", "other": "thraskin"}
    identity.update({f"dup{i}": "thraskin" for i in range(10)})
    out["unique_information@5 under flooding"] = float(
        unique_information_at_k(order, identity, 5)
    )

    # rank of the gold among 64 hard negatives.
    facts = [
        Fact(gold_id="gold", text="gold the kelmady arrangement", vector=at_cosine(0.90, off=1))
    ]
    facts += [
        Fact(gold_id=f"h{i}", text=f"h{i} near miss", vector=v)
        for i, v in enumerate(band(0.85, 64, first_off=2))
    ]
    order = await run("rep-hard", facts, ("gold",), 70)
    out["rank of gold among 64 hard negatives"] = float(
        order.index("gold") + 1 if "gold" in order else 999
    )
    return out


def _table(measured: dict[str, float]) -> str:
    width = max(len(k) for k in FLOORS)
    lines = [
        "",
        "  CAPABILITY LEDGER",
        "  " + "-" * (width + 26),
        f"  {'metric':<{width}}  {'measured':>9}  {'floor':>7}  ok",
        "  " + "-" * (width + 26),
    ]
    for name, floor in FLOORS.items():
        value = measured.get(name, float("nan"))
        # The rank metric is the one where LOWER is better.
        ok = value <= floor if name.startswith("rank of") else value >= floor
        lines.append(f"  {name:<{width}}  {value:9.3f}  {floor:7.3f}  {'.' if ok else 'X'}")
    lines.append("  " + "-" * (width + 26))
    return "\n".join(lines)


async def test_the_capability_ledger_meets_its_floors(geometry: tuple) -> None:
    """One assertion per metric, with the full table printed either way.

    Printed on success too (`-s` to see it), because the point of a ledger is to
    be read, not only to gate.
    """
    measured = await _measure(geometry)
    print(_table(measured))

    failures = []
    for name, floor in FLOORS.items():
        value = measured.get(name)
        assert value is not None, f"{name} was not measured"
        ok = value <= floor if name.startswith("rank of") else value >= floor
        if not ok:
            failures.append(f"{name}: {value:.3f} vs floor {floor:.3f}")
    assert not failures, (
        "capability floors breached:\n  " + "\n  ".join(failures) + "\n" + _table(measured)
    )


async def test_every_floor_is_actually_measured(geometry: tuple) -> None:
    """A floor with no measurement is a floor that cannot fail.

    Guards the failure mode where a metric is renamed in `_measure` and its entry
    in `FLOORS` silently stops being checked -- the assertion above would still
    pass on the remaining ones.
    """
    measured = await _measure(geometry)
    missing = set(FLOORS) - set(measured)
    assert not missing, f"floors with no measurement: {sorted(missing)}"
    extra = set(measured) - set(FLOORS)
    assert not extra, f"measurements with no floor -- add them to FLOORS: {sorted(extra)}"


def test_no_floor_is_vacuous() -> None:
    """A guard on the guards, matching the minimal-pair battery's.

    A floor of 0.0 on a recall metric passes unconditionally. If a capability
    genuinely cannot be served, it belongs in a named test explaining why -- the
    way `direction` is handled in the minimal-pair file -- not as a zero in a
    table that looks like coverage.
    """
    for name, floor in FLOORS.items():
        if name.startswith("rank of"):
            continue
        assert floor > 0.0, f"{name} has a vacuous floor of {floor}"
