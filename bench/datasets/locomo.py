"""LoCoMo: very-long-term multi-session dialogue.

Ten dialogues between two speakers, 1,986 questions, evidence labelled at
TURN granularity (`dia_id`) — which is what makes objective retrieval scoring
possible without an LLM judge.

Category 5 is adversarial (unanswerable) and is mapped to `is_abstention`. A
system that confidently answers the unanswerable is worse than one that
declines, so it is scored separately from accuracy throughout.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .base import Corpus, Dataset, Document, Question

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
ADVERSARIAL = 5
_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")


def _parse_date(raw: str | None) -> datetime:
    if raw:
        match = _DATE_RE.search(raw)
        if match:
            day, month, year = match.groups()
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    return datetime.strptime(f"{day} {month} {year}", fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
    return datetime(2023, 1, 1, tzinfo=UTC)


class LoCoMo(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return "locomo"

    @property
    def evidence_granularity(self) -> str:
        return "turn"

    def load(self, *, limit_corpora: int | None = None) -> list[Corpus]:
        raw = json.loads(self.path.read_text())
        corpora: list[Corpus] = []
        for sample in raw[: limit_corpora or len(raw)]:
            sample_id = str(sample.get("sample_id", f"conv{len(corpora)}"))
            corpus = Corpus(corpus_id=sample_id)
            blob = sample.get("conversation", {}) or {}

            keys = sorted(
                (k for k in blob if k.startswith("session_") and isinstance(blob[k], list)),
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
            )
            for key in keys:
                when = _parse_date(blob.get(f"{key}_date_time"))
                for turn in blob[key]:
                    text = (turn.get("text") or "").strip()
                    dia_id = turn.get("dia_id")
                    if not text or not dia_id:
                        continue
                    speaker = str(turn.get("speaker", "Unknown"))
                    corpus.documents.append(
                        Document(
                            id=str(dia_id),
                            text=f"{speaker}: {text}",
                            occurred_at=when,
                            group=key,
                            speaker=speaker,
                        )
                    )

            for index, item in enumerate(sample.get("qa", []) or []):
                question = (item.get("question") or "").strip()
                if not question:
                    continue
                evidence = item.get("evidence") or []
                if isinstance(evidence, str):
                    evidence = [evidence]
                category = int(item.get("category", 0) or 0)
                corpus.questions.append(
                    Question(
                        qid=f"{sample_id}::q{index}",
                        corpus_id=sample_id,
                        text=question,
                        answer=str(
                            item.get("answer", item.get("adversarial_answer", ""))
                        ).strip(),
                        evidence_ids=tuple(str(e) for e in evidence),
                        category=CATEGORY_NAMES.get(category, f"category_{category}"),
                        is_abstention=category == ADVERSARIAL,
                    )
                )
            corpora.append(corpus)
        return corpora


__all__ = ["LoCoMo"]
