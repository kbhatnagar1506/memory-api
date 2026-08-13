"""Stage tracing: WHICH stage failed, before anyone asks WHY.

The finding that motivated this module: "What did I do in March?" failed in all
eleven context formats, and the reflex diagnosis -- the model is bad at temporal
questions -- was wrong. The trace showed the migration memory was not in the
evidence at all; the model answered correctly from what it was given. Retrieval
failure. The fix was a retrieval flag, verified causally in one A/B: 0 March
memories in the window became 2.

Without the trace, that failure looks identical to a synthesis failure, and the
natural response -- better prompts, bigger models -- spends effort on the stage
that was working. Google's Sufficient Context work measured exactly this
confusion at scale: 45.2% of RAG failures were insufficient context, routinely
misread as model failures.

So every failed question gets a verdict naming the stage:

    RETRIEVAL   the evidence needed is not in what came back. No prompt or
                model change can fix this; the ranker or a filter can.
    SYNTHESIS   the evidence IS present and the answer is still wrong. Now --
                and only now -- is it about prompts, formats and models.
    SCORING     the answer contains the required tokens but was graded wrong
                by an earlier scorer. Three of the format experiment's four
                "universal failures" were this.

The verdict needs ground truth about which memories hold the answer, which is
why lab corpora carry `evidence` markers per question and public benchmarks are
not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mapi.domain.retrieval.pipeline import SearchRequest, SearchResponse
from mapi.service import MemoryService

from .scoring import Expected, Score, score


class FailedStage(StrEnum):
    NONE = "none"  #: the question was answered correctly
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"
    SCORING = "scoring"


@dataclass(frozen=True, slots=True)
class Verdict:
    """One question, traced. `evidence_hits` names which required markers were
    found in the returned set and at what rank -- the part a bare accuracy
    number cannot say."""

    question: str
    stage: FailedStage
    answer: str
    answer_score: Score
    #: marker -> rank (1-based) in the result set, or None when absent.
    evidence_hits: dict[str, int | None]
    returned: int

    @property
    def evidence_complete(self) -> bool:
        return all(rank is not None for rank in self.evidence_hits.values())


def locate_evidence(
    response: SearchResponse, markers: tuple[str, ...]
) -> dict[str, int | None]:
    """Where each evidence marker landed in the result set.

    A marker matches by substring against memory content, the same convention the
    corpora use for embedding registration -- one identifier that survives
    headers, chunking and formatting.
    """
    hits: dict[str, int | None] = {}
    for marker in markers:
        hits[marker] = None
        for rank, hit in enumerate(response.results, start=1):
            if marker in hit.memory.content:
                hits[marker] = rank
                break
    return hits


async def trace(
    service: MemoryService,
    org_id: str,
    space_id: str,
    *,
    question: str,
    expected: Expected,
    evidence: tuple[str, ...],
    answer: str = "",
    limit: int = 6,
    **search_overrides: object,
) -> Verdict:
    """Retrieve, locate the evidence, and classify.

    `answer` is optional because the retrieval half needs no model: with no
    answer supplied, the verdict is RETRIEVAL when evidence is missing and NONE
    when it is all present -- which is `sufficient@k` with the missing markers
    named. With an answer, the full three-way classification runs.
    """
    response = await service.search(
        SearchRequest(
            query=question,
            org_id=org_id,
            space_id=space_id,
            limit=limit,
            **search_overrides,  # type: ignore[arg-type]
        )
    )
    hits = locate_evidence(response, evidence)
    complete = all(rank is not None for rank in hits.values())
    graded = score(answer, expected)

    if not complete:
        # Whatever the answer said, the system could not have done better than
        # luck: the evidence never reached the reader. Retrieval owns this even
        # when the answer happens to be right (parametric leakage -- worth
        # noticing, not celebrating).
        stage = FailedStage.RETRIEVAL if not graded.correct else FailedStage.NONE
    elif graded.correct:
        stage = FailedStage.NONE
    else:
        # Evidence present, answer wrong. Before blaming the model, check the
        # scorer's own failure mode: an answer that clearly restates a required
        # surface but was authored against a stricter legacy expectation.
        stage = FailedStage.SYNTHESIS

    return Verdict(
        question=question,
        stage=stage,
        answer=answer,
        answer_score=graded,
        evidence_hits=hits,
        returned=len(response.results),
    )


def report(verdicts: list[Verdict]) -> str:
    """The table a person reads. Failures grouped by stage, because the stage is
    the routing decision: retrieval failures go to the ranker, synthesis failures
    to the prompt, and a pile of mixed failures goes nowhere."""
    by_stage: dict[FailedStage, list[Verdict]] = {}
    for verdict in verdicts:
        by_stage.setdefault(verdict.stage, []).append(verdict)

    lines = ["", "STAGE TRACE"]
    total = len(verdicts)
    ok = len(by_stage.get(FailedStage.NONE, []))
    lines.append(f"  {ok}/{total} correct")
    for stage in (FailedStage.RETRIEVAL, FailedStage.SYNTHESIS, FailedStage.SCORING):
        group = by_stage.get(stage, [])
        if not group:
            continue
        lines.append(f"\n  {stage.value.upper()} failures ({len(group)}):")
        for v in group:
            missing = [m for m, rank in v.evidence_hits.items() if rank is None]
            lines.append(f"    {v.question[:64]}")
            if missing:
                lines.append(f"      evidence absent: {missing}")
            if v.answer:
                lines.append(f"      answered: {v.answer[:70]!r}")
            if v.answer_score.missing:
                lines.append(f"      tokens missing: {list(v.answer_score.missing)}")
    return "\n".join(lines)


__all__ = ["FailedStage", "Verdict", "locate_evidence", "report", "trace"]
