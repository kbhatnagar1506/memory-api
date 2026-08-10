"""Read-time synthesis: derive answers that exist in no single memory.

Measured motivation (LongMemEval, n=500): retrieval delivers complete evidence
for 97.2% of questions, yet 45% of the remaining failures are COUNT questions
— "how many bikes do I own?" answered "Multiple" with all three bikes in
context. The model was never missing memory; it was being asked to be a
calculator over 2,400 tokens of prose.

Read-time extraction is not a replacement for write-time extraction; both
ship, and the split is deliberate. `extract.py` decomposes a document into
atomic claims as it arrives, which is what makes a multi-fact passage
retrievable fact-by-fact instead of as one averaged vector. What that cannot
do is compute — no write-time pass knows that a question about totals is
coming — and the write path stays LOSSLESS either way: extracted claims are
ADDED beside the original, never in place of it, so nothing is decided about
what matters before the question exists.

The measured position on the trade-off, since the numbers are ours: on
LongMemEval, write-time extraction moved answer accuracy 0.826 -> 0.781
overall, and the sign split cleanly by memory kind. Every semantic
capability held or improved; every episodic one lost. Claims answer "what is
true"; episodes answer "what happened". That is why extraction is opt-in per
request rather than a default, and why `route_by_kind` exists.

    map     one extraction call per retrieved memory, in parallel
    ground  drop any row whose quote is not in its source  (code, not judge)
    reduce  count/order/span in code; only COMPARE composes via the model
"""

from .classify import QuestionKind, classify
from .derive import DerivedAnswer, Extraction, derive_answer, ground
from .scope import DateRange, extract_scope

__all__ = [
    "DateRange",
    "DerivedAnswer",
    "Extraction",
    "QuestionKind",
    "classify",
    "derive_answer",
    "extract_scope",
    "ground",
]
