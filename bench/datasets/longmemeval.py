"""LongMemEval: chat-assistant memory across five abilities.

500 questions, each with its OWN ~115K-token haystack of ~50 sessions. That
per-question haystack is the structural difference from LoCoMo and the reason
`Corpus` is the unit of ingestion: each instance becomes one corpus holding
exactly one question.

Evidence is labelled at SESSION granularity (`answer_session_ids`), so
documents here are sessions, not turns. Session-level recall is a strictly
easier target than LoCoMo's turn-level recall and the two must never be
compared directly — `evidence_granularity` exists so a report can say so.

The capability we care most about is `knowledge-update` (78 questions): it
tests whether a system tracks a fact being replaced, which is precisely what
our supersession graph is for and what LoCoMo barely probes.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .base import Corpus, Dataset, Document, Question

#: LongMemEval marks abstention questions with this suffix.
_ABSTENTION_SUFFIX = "_abs"
_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def _parse_date(raw: str | None) -> datetime:
    if raw:
        match = _DATE_RE.search(raw)
        if match:
            year, month, day = (int(g) for g in match.groups())
            try:
                return datetime(year, month, day, tzinfo=UTC)
            except ValueError:
                pass
    return datetime(2023, 1, 1, tzinfo=UTC)


class LongMemEval(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return "longmemeval"

    @property
    def evidence_granularity(self) -> str:
        return "session"

    def load(self, *, limit_corpora: int | None = None) -> list[Corpus]:
        raw = json.loads(self.path.read_text())
        if limit_corpora:
            raw = _stratified(raw, limit_corpora)

        corpora: list[Corpus] = []
        for instance in raw:
            qid = str(instance.get("question_id", f"q{len(corpora)}"))
            sessions = instance.get("haystack_sessions") or []
            session_ids = instance.get("haystack_session_ids") or []
            dates = instance.get("haystack_dates") or []

            corpus = Corpus(corpus_id=qid)
            for position, turns in enumerate(sessions):
                session_id = str(
                    session_ids[position] if position < len(session_ids) else f"s{position}"
                )
                when = _parse_date(dates[position] if position < len(dates) else None)
                # One document per session: that is the granularity the
                # benchmark labels, so retrieval is scored on the same unit it
                # is asked about.
                body = "\n".join(
                    f"{t.get('role', '?')}: {(t.get('content') or '').strip()}"
                    for t in turns
                    if (t.get("content") or "").strip()
                )
                if not body:
                    continue
                corpus.documents.append(
                    Document(
                        id=session_id,
                        text=body,
                        occurred_at=when,
                        group=session_id,
                        metadata={"session_index": str(position)},
                    )
                )

            category = str(instance.get("question_type", "unknown"))
            corpus.questions.append(
                Question(
                    qid=qid,
                    corpus_id=qid,
                    text=str(instance.get("question", "")).strip(),
                    answer=str(instance.get("answer", "")).strip(),
                    evidence_ids=tuple(
                        str(s) for s in (instance.get("answer_session_ids") or [])
                    ),
                    category=category,
                    is_abstention=qid.endswith(_ABSTENTION_SUFFIX),
                    asked_at=_parse_date(instance.get("question_date")),
                )
            )
            corpora.append(corpus)
        return corpora


def _stratified(raw: list[dict], limit: int) -> list[dict]:
    """Even coverage across question types, so a small run still measures every
    capability rather than whichever type happens to sort first."""
    buckets: dict[str, list[dict]] = {}
    for item in raw:
        buckets.setdefault(str(item.get("question_type", "unknown")), []).append(item)
    per = max(1, limit // max(len(buckets), 1))
    picked: list[dict] = []
    for key in sorted(buckets):
        picked.extend(buckets[key][:per])
    return picked[:limit]


__all__ = ["LongMemEval"]
