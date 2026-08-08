"""The benchmark dataset interface.

Every memory benchmark tests something different — LoCoMo tests multi-session
recall, LongMemEval tests knowledge updates and abstention, BEAM tests scale —
so a single benchmark score is both incomplete and gameable. This protocol lets
one harness run all of them and report PER CAPABILITY.

Deliberately not provided: a blended composite score. Collapsing distinct
capabilities into one number is exactly the artefact that discredited vendor
benchmark marketing in this category; a system can be excellent at recall and
dangerous at abstention, and one number hides that.

Two granularities of evidence are supported because the datasets differ:
LoCoMo labels the individual dialogue turn, LongMemEval labels the session. The
harness scores whatever `evidence_ids` refer to, so `Document.id` must be at
the same granularity the dataset labels.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Document:
    """One ingestible unit, at the granularity the benchmark labels evidence."""

    id: str
    text: str
    occurred_at: datetime
    #: Groups documents that belong together (a session, a conversation).
    group: str = ""
    speaker: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    #: Which corpus this question is asked against.
    corpus_id: str
    text: str
    answer: str
    #: Document ids that support the answer. Empty for unanswerable questions.
    evidence_ids: tuple[str, ...]
    #: Benchmark-native capability label, reported verbatim. Not normalized
    #: across benchmarks: "multi-session" in LongMemEval and "multi_hop" in
    #: LoCoMo are related but not the same thing, and pretending otherwise
    #: would invent a comparison the data does not support.
    category: str
    #: True when the correct behaviour is to decline.
    is_abstention: bool = False
    #: Answer-time reference point, where the benchmark supplies one.
    asked_at: datetime | None = None


@dataclass(slots=True)
class Corpus:
    """One haystack plus the questions asked against it.

    LoCoMo shares a corpus across ~200 questions; LongMemEval gives every
    question its own ~115K-token haystack. Modelling the corpus as the unit
    rather than the question is what lets one harness serve both without
    re-ingesting per question.
    """

    corpus_id: str
    documents: list[Document] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


class Dataset(abc.ABC):
    """A benchmark. Loads corpora; the harness does everything else."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def evidence_granularity(self) -> str:
        """ "turn" or "session" — what `evidence_ids` refer to. Reported so a
        reader never compares a turn-level recall to a session-level one."""

    @abc.abstractmethod
    def load(self, *, limit_corpora: int | None = None) -> list[Corpus]: ...

    @staticmethod
    def available(path: str) -> bool:
        from pathlib import Path

        return Path(path).exists()


__all__ = ["Corpus", "Dataset", "Document", "Question"]
