"""Extraction recall: what does the map stage MISS?

The most dangerous unmeasured number in the system. Everything downstream of
extraction inherits its recall: miss one of three bikes and code counts two --
confidently, with provenance, and wrong. Derivation is currently restricted to
filling declines precisely because this number did not exist.

Ground truth without hand-labelling: synthesise documents containing a KNOWN
number of instances, then ask the real map stage to find them. Recall is exact
because we wrote the documents. Distractors and paraphrase are included so the
probe measures finding, not pattern-matching.

    python -m bench.probe_extraction

Reports recall (found / planted), precision-after-grounding, and the
fabrication rate. A low recall means derive must stay advisory; a high one
means it can be trusted to override.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from supermemory.domain.synthesis import QuestionKind
from supermemory.domain.synthesis.derive import SourceDoc, derive_answer

from .harness import build_model_client


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    kind: QuestionKind
    #: Documents, each with the number of target instances it contains.
    docs: tuple[tuple[str, int], ...]

    @property
    def planted(self) -> int:
        return sum(n for _, n in self.docs)


#: Deliberately varied: instances stated plainly, buried mid-paragraph, spread
#: across documents, and surrounded by near-miss distractors.
CASES: tuple[Case, ...] = (
    Case(
        "How many bikes do I own?",
        QuestionKind.COUNT,
        (
            (
                "I picked up a road bike today. My old mountain bike is still in the shed. "
                "My neighbour also rides a bike but that one is not mine.",
                2,
            ),
            ("The e-bike I ordered finally arrived, so the garage is full now.", 1),
        ),
    ),
    Case(
        "How many times did I go to the gym?",
        QuestionKind.COUNT,
        (
            (
                "Monday I went to the gym before work. Wednesday I skipped it because of "
                "the rain. Friday I made it to the gym again for a leg session.",
                2,
            ),
            (
                "Went to the gym Sunday morning, quiet and empty. I also thought about "
                "going Saturday but did not.",
                1,
            ),
        ),
    ),
    Case(
        "How many books did I finish?",
        QuestionKind.COUNT,
        (
            (
                "Finished The Nightingale on the train. Started Dune but only got "
                "forty pages in.",
                1,
            ),
            (
                "Finally finished Dune this weekend. Also finished a short story "
                "collection I had been picking at for months.",
                2,
            ),
        ),
    ),
    Case(
        "How many doctor appointments did I have?",
        QuestionKind.COUNT,
        (
            (
                "Saw the dentist on the 3rd. My GP appointment was on the 11th. "
                "I need to book an optician but have not yet.",
                2,
            ),
            ("The follow-up with my GP happened Thursday.", 1),
        ),
    ),
    Case(
        "How many plants did I buy?",
        QuestionKind.COUNT,
        (
            (
                "Bought a monstera and a snake plant at the market. My sister gave me "
                "a cutting but I did not buy that one.",
                2,
            ),
            ("Picked up a fiddle leaf fig on the way home.", 1),
        ),
    ),
)


async def main() -> None:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    client = build_model_client("gemini-2.5-flash", project)

    async def complete(prompt: str) -> str:
        text, _ = await client.complete(prompt, max_tokens=1024)
        return text

    total_planted = total_found = total_rejected = 0
    errors: list[int] = []
    print(f"{'case':44s} {'planted':>8s} {'found':>6s} {'recall':>7s} {'fabricated':>11s}")
    print("-" * 82)
    for case in CASES:
        docs = [
            SourceDoc(id=f"d{i}", text=text, occurred_at=datetime(2026, 3, i + 1, tzinfo=UTC))
            for i, (text, _) in enumerate(case.docs)
        ]
        result = await derive_answer(case.question, case.kind, docs, complete)
        found = len(result.table) if result else 0
        rejected = result.rejected if result else 0
        total_planted += case.planted
        total_found += found
        total_rejected += rejected
        errors.append(abs(found - case.planted))
        recall = found / case.planted if case.planted else 0.0
        print(
            f"{case.question[:44]:44s} {case.planted:8d} {found:6d} "
            f"{recall:6.0%} {rejected:11d}"
        )

    print("-" * 82)
    recall = total_found / total_planted if total_planted else 0.0
    produced = total_found + total_rejected
    fabrication = total_rejected / produced if produced else 0.0
    print(
        f"{'TOTAL':44s} {total_planted:8d} {total_found:6d} {recall:6.0%} {total_rejected:11d}"
    )
    print()
    # Net recall is the WRONG headline: one case over-extracting cancels
    # another under-extracting and reports a flattering 100%. What a
    # downstream count actually inherits is the per-case error.
    mae = sum(errors) / len(errors) if errors else 0.0
    exact = sum(1 for e in errors if e == 0)
    print(f"  net recall            : {recall:.1%}  (MISLEADING — over/under cancel)")
    print(f"  exactly right         : {exact}/{len(errors)} cases")
    print(f"  mean absolute error   : {mae:.2f} instances per case")
    print(f"  fabrication rate      : {fabrication:.1%}  (rows whose quote was not in source)")
    print()
    print("  Recall bounds every derived answer: a miss becomes an undercount")
    print("  delivered confidently, with provenance. Below ~90% derivation")
    print("  should stay advisory (fill declines only) rather than overriding.")


if __name__ == "__main__":
    asyncio.run(main())
