"""LoCoMo dataset loading and evaluation metrics.

LoCoMo is long-term conversational memory: ten multi-session dialogues between
two speakers, with 1,986 questions whose ground truth includes the `dia_id` of
every turn that supports the answer. Those evidence labels are what make an
*objective* retrieval score possible — recall, MRR and nDCG are computed against
labelled turns, with no LLM in the loop and therefore no judge variance.

Question categories, per the LoCoMo paper:

    1  multi-hop      answer requires combining several turns
    2  temporal       answer is a date or ordering
    3  open-domain    requires world knowledge plus the dialogue
    4  single-hop     answer is in one turn
    5  adversarial    unanswerable from the dialogue

Category 5 is scored separately throughout. A system that answers an
unanswerable question confidently is worse than one that declines, and folding
those into a single accuracy number hides exactly that behaviour.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
ADVERSARIAL = 5


@dataclass(frozen=True, slots=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    session: str
    session_date: str

    @property
    def content(self) -> str:
        """Speaker-attributed text. Who said it is part of the fact."""
        return f"{self.speaker}: {self.text}"


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    sample_id: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    category: int

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category_{self.category}")

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL


@dataclass(slots=True)
class Conversation:
    sample_id: str
    turns: list[Turn] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")


def _parse_date(raw: str | None) -> datetime:
    """LoCoMo session dates look like '7 May, 2023 1:00 pm'.

    Falls back to a fixed epoch rather than raising: a missing date must not
    stop a benchmark run, and recency decay is measured separately anyway.
    """
    if raw:
        match = _DATE_RE.search(raw)
        if match:
            day, month_name, year = match.groups()
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    return datetime.strptime(f"{day} {month_name} {year}", fmt).replace(
                        tzinfo=UTC
                    )
                except ValueError:
                    continue
    return datetime(2023, 1, 1, tzinfo=UTC)


def load(path: Path, *, limit_conversations: int | None = None) -> list[Conversation]:
    raw = json.loads(Path(path).read_text())
    conversations: list[Conversation] = []

    for sample in raw[: limit_conversations or len(raw)]:
        sample_id = str(sample.get("sample_id", f"conv{len(conversations)}"))
        convo = Conversation(sample_id=sample_id)
        blob = sample.get("conversation", {}) or {}

        session_keys = sorted(
            (k for k in blob if k.startswith("session_") and isinstance(blob[k], list)),
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
        )
        for key in session_keys:
            date = _parse_date(blob.get(f"{key}_date_time"))
            for turn in blob[key]:
                text = (turn.get("text") or "").strip()
                dia_id = turn.get("dia_id")
                if not text or not dia_id:
                    continue
                convo.turns.append(
                    Turn(
                        dia_id=str(dia_id),
                        speaker=str(turn.get("speaker", "Unknown")),
                        text=text,
                        session=key,
                        session_date=date.isoformat(),
                    )
                )

        for index, item in enumerate(sample.get("qa", []) or []):
            question = (item.get("question") or "").strip()
            if not question:
                continue
            evidence = item.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            convo.questions.append(
                Question(
                    qid=f"{sample_id}::q{index}",
                    sample_id=sample_id,
                    question=question,
                    # Adversarial items carry an explanatory string; others may
                    # carry a number. Normalize to text.
                    answer=str(item.get("answer", item.get("adversarial_answer", ""))).strip(),
                    evidence=tuple(str(e) for e in evidence),
                    category=int(item.get("category", 0) or 0),
                )
            )
        conversations.append(convo)
    return conversations


# -- retrieval metrics ---------------------------------------------------------
#
# Computed against LoCoMo's labelled evidence turns. No LLM, so these numbers
# are exactly reproducible and carry no judge variance.


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of evidence turns present in the top k."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any evidence turn is in the top k. The 'could it possibly answer' bar."""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first evidence turn."""
    if not relevant:
        return 0.0
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting evidence high, not merely present."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


__all__ = [
    "ADVERSARIAL",
    "CATEGORY_NAMES",
    "Conversation",
    "Question",
    "Turn",
    "hit_at_k",
    "load",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
