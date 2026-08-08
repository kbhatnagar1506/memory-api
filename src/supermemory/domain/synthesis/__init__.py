"""Read-time synthesis: derive answers that exist in no single memory.

Measured motivation (LongMemEval, n=500): retrieval delivers complete evidence
for 97.2% of questions, yet 45% of the remaining failures are COUNT questions
— "how many bikes do I own?" answered "Multiple" with all three bikes in
context. The model was never missing memory; it was being asked to be a
calculator over 2,400 tokens of prose.

The competing fix — extracting facts at write time, as Mem0/Zep/Letta do — is
guessing: it decides what matters before any question exists, and the
independent cost study (arXiv 2603.04814) measured the recall it loses.
Extraction at read time inverts that: the question says exactly what to
extract, code does the arithmetic, and the lossless write path keeps every
future question answerable.

    map     one extraction call per retrieved memory, in parallel
    ground  drop any row whose quote is not in its source  (code, not judge)
    reduce  count/order/span in code; only COMPARE composes via the model
"""

from .classify import QuestionKind, classify
from .derive import DerivedAnswer, Extraction, derive_answer, ground

__all__ = [
    "DerivedAnswer",
    "Extraction",
    "QuestionKind",
    "classify",
    "derive_answer",
    "ground",
]
