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


#: LoCoMo evidence is mostly a list of dia_ids, but a handful of entries pack
#: several into one string ("D8:6; D9:17") and a few name turns that do not
#: exist in the transcript. Both silently score 0 on full recall forever: the
#: id can never match a retrieved document, so the question is unwinnable no
#: matter what retrieval does. 9 of 1,986 questions are affected (0.5%).
_EVIDENCE_SPLIT = re.compile(r"[;,]\s*")


def _clean_evidence(raw: object, known_ids: set[str]) -> tuple[str, ...]:
    """Split packed ids and drop ones with no matching turn.

    Dropping is the honest choice over keeping: a label pointing at a turn that
    does not exist measures the dataset, not the retriever. Questions left with
    NO resolvable evidence are excluded from retrieval scoring by the harness,
    which skips questions with empty `evidence_ids`.
    """
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for item in items:
        for part in _EVIDENCE_SPLIT.split(str(item)):
            candidate = part.strip()
            if candidate and candidate in known_ids and candidate not in out:
                out.append(candidate)
    return tuple(out)


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

            known_ids = {d.id for d in corpus.documents}
            for index, item in enumerate(sample.get("qa", []) or []):
                question = (item.get("question") or "").strip()
                if not question:
                    continue
                evidence = _clean_evidence(item.get("evidence"), known_ids)
                category = int(item.get("category", 0) or 0)
                corpus.questions.append(
                    Question(
                        qid=f"{sample_id}::q{index}",
                        corpus_id=sample_id,
                        text=question,
                        answer=str(
                            item.get("answer", item.get("adversarial_answer", ""))
                        ).strip(),
                        evidence_ids=evidence,
                        category=CATEGORY_NAMES.get(category, f"category_{category}"),
                        is_abstention=category == ADVERSARIAL,
                    )
                )
            corpora.append(corpus)
        return corpora


__all__ = ["LoCoMo"]
