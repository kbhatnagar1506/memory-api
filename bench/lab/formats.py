"""Every way we can hand retrieved evidence to a model, as one function each.

WHY THIS EXISTS. Retrieval is at `full_recall@k` 0.968 and accuracy is ~82.4%,
so complete evidence reaches the model for 97.2% of questions and the model
still gets ~15% of them wrong. The gap is on the answering side, and of
everything on that side, the RENDERING of the evidence is the one variable
never systematically varied. Every arm so far changed what was retrieved or
which model read it. This changes only how the same bytes are laid out.

The axes, because "all the formats" is four independent questions and not one:

  DELIMITER    how one record is separated from the next
  ENRICHMENT   what is stated about a record besides its text
  REDUCTION    whether the model reads sessions or extracted claims
  POSITION     where the question sits relative to the evidence

Each renderer takes the same list of records and returns a string. Nothing here
retrieves, scores or decides -- so a difference in outcome between two of these
is attributable to layout alone, which is the only reason the comparison means
anything.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class Record:
    """One retrieved unit, with everything any renderer might want to state."""

    n: int
    text: str
    occurred_at: datetime
    score: float = 0.0
    source: str = ""
    session_id: str = ""

    @property
    def day(self) -> date:
        return self.occurred_at.date()

    def age_days(self, asked_at: date) -> int:
        return (asked_at - self.day).days


Renderer = Callable[[Sequence[Record], date], str]


def _iso(record: Record) -> str:
    """Minute precision, not day.

    Three LongMemEval questions have two gold sessions on the SAME day. A
    date-only header leaves them positionally unorderable, so "which happened
    first" becomes unanswerable from evidence the model cannot sequence.
    """
    return record.occurred_at.isoformat(timespec="minutes")


def _relative(record: Record, asked_at: date) -> str:
    """ "28 days ago", in the units a person would use.

    Stated rather than left as arithmetic because 25 of 133 temporal questions
    ask "how many weeks ago" and the subtraction is where they fail, not the
    retrieval.
    """
    days = record.age_days(asked_at)
    if days < 0:
        return "in the future"
    if days == 0:
        return "today"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days} days ago ({days // 7} weeks)"
    return f"{days} days ago ({days // 30} months)"


# -- DELIMITER axis: same content, different record boundaries -------------


def blocks(records: Sequence[Record], asked_at: date) -> str:
    """BASELINE. What the harness ships: a timestamp header, `---` between.

    Every number in every arm so far was measured with this, so it is the
    reference and not a candidate.
    """
    return "\n\n---\n\n".join(f"[{_iso(r)}]\n{r.text}" for r in records)


def numbered(records: Sequence[Record], asked_at: date) -> str:
    """Ordinals, so the model has a handle it can cite.

    Hypothesis: a model that can say "from excerpt 3" commits to a source, and
    a commitment is checkable. Unnumbered evidence lets it blend excerpts
    without ever naming one, which is the shape most of our wrong answers have.
    """
    return "\n\n".join(f"### Excerpt {r.n} — {_iso(r)}\n{r.text}" for r in records)


def xml(records: Sequence[Record], asked_at: date) -> str:
    """Tagged blocks. The most unambiguous boundary marker there is.

    Hypothesis: prose containing `---` or `###` can be confused with the
    delimiter; prose containing `</excerpt>` essentially never is. Attributes
    also put the metadata somewhere the text cannot be mistaken for it.
    """
    parts = [f'<excerpt n="{r.n}" date="{_iso(r)}">\n{r.text}\n</excerpt>' for r in records]
    return "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>"


def json_array(records: Sequence[Record], asked_at: date) -> str:
    """A JSON array of objects.

    Hypothesis, and it cuts both ways: the structure is unmistakable, but
    conversational prose inside a JSON string is full of escaped quotes and
    `\\n`, and every one of those is a token spent on syntax plus a chance to
    misread a newline as literal text.
    """
    return json.dumps(
        [{"n": r.n, "date": _iso(r), "text": r.text} for r in records],
        indent=1,
        ensure_ascii=False,
    )


def yaml_list(records: Sequence[Record], asked_at: date) -> str:
    """YAML block scalars: JSON's structure without JSON's escaping.

    Hypothesis: this is the format that gets structure for free. `|` block
    scalars carry multi-line prose verbatim, so there is no escaping tax.
    """
    out = []
    for r in records:
        body = "\n".join(f"    {line}" for line in r.text.strip().splitlines())
        out.append(f"- n: {r.n}\n  date: {_iso(r)}\n  text: |\n{body}")
    return "\n".join(out)


def markdown_table(records: Sequence[Record], asked_at: date) -> str:
    """One row per record. The most compact layout available.

    Hypothesis: compactness is the wrong goal here and this will lose. A table
    cell cannot hold a paragraph without newline-flattening, and flattening a
    session destroys the turn structure the answer often depends on. Included
    BECAUSE it is the format people reach for first.
    """
    rows = [f"| {r.n} | {r.day.isoformat()} | {' '.join(r.text.split())} |" for r in records]
    return "| # | date | excerpt |\n|---|---|---|\n" + "\n".join(rows)


def key_value(records: Sequence[Record], asked_at: date) -> str:
    """Named fields per record, no nesting, no syntax to parse.

    Hypothesis: the cheapest way to make metadata unmissable. Everything a
    renderer could say is said as `field: value`, which needs no format
    knowledge at all.
    """
    out = []
    for r in records:
        out.append(
            f"--- excerpt {r.n} ---\n"
            f"date: {_iso(r)}\n"
            f"when: {_relative(r, asked_at)}\n"
            f"text:\n{r.text.strip()}"
        )
    return "\n\n".join(out)


# -- ENRICHMENT axis: same delimiters, more stated per record --------------


def timeline(records: Sequence[Record], asked_at: date) -> str:
    """Baseline blocks plus the elapsed time, computed.

    Hypothesis, and the most specific one here: "how many weeks ago" fails
    because the model is asked to subtract dates, and stating the answer to
    that subtraction next to each record removes the arithmetic entirely. If
    this beats `blocks` on temporal questions, the temporal problem was never
    retrieval.
    """
    header = f"Today is {asked_at.isoformat()}. Events oldest first.\n"
    body = "\n\n".join(f"[{_iso(r)} — {_relative(r, asked_at)}]\n{r.text}" for r in records)
    return header + "\n" + body


def with_provenance(records: Sequence[Record], asked_at: date) -> str:
    """Blocks plus retrieval rank and score.

    Hypothesis: this HURTS, and it is here to be falsified. A score tells the
    model which excerpt the retriever liked, which is not which excerpt holds
    the answer -- our own failures are mostly answers sitting at rank 4. Making
    rank visible invites anchoring on rank 1.
    """
    return "\n\n---\n\n".join(
        f"[{_iso(r)}] rank {r.n} score {r.score:.3f}\n{r.text}" for r in records
    )


def newest_first(records: Sequence[Record], asked_at: date) -> str:
    """Baseline, reversed.

    Hypothesis: for "which value is current", the current one arrives first
    instead of last. Against that, recency-first fights the "if a fact CHANGED,
    take the most recent" instruction in the prompt, which assumes oldest-first
    order. One of those two effects is bigger and this is how we find out.
    """
    return blocks(list(reversed(list(records))), asked_at)


# -- REDUCTION axis: fewer bytes, more processing done upstream ------------


def claims_only(records: Sequence[Record], asked_at: date) -> str:
    """Sentence-per-line, the shape `derive` builds its fact table in.

    Hypothesis: this is the arm most likely to lose, and it is the one worth
    measuring because it is what "better memory" usually means. Extraction
    already cost us -21 to -54 questions once: `full_recall@k` fell 0.968 ->
    0.948 while MRR rose, i.e. sharper retrieval and worse coverage. Splitting
    to sentences here is the same trade without the retrieval half -- the
    answer is present either way, so any loss is the model failing to read
    fragments it could read as prose.
    """
    out = []
    for r in records:
        sentences = [s.strip() for s in r.text.replace("\n", " ").split(". ") if s.strip()]
        for sentence in sentences:
            out.append(f"- [{r.day.isoformat()}] {sentence.rstrip('.')}.")
    return "\n".join(out)


#: Every renderer, with the baseline first so it prints first.
RENDERERS: dict[str, Renderer] = {
    "blocks": blocks,
    "numbered": numbered,
    "xml": xml,
    "json": json_array,
    "yaml": yaml_list,
    "markdown_table": markdown_table,
    "key_value": key_value,
    "timeline": timeline,
    "with_provenance": with_provenance,
    "newest_first": newest_first,
    "claims_only": claims_only,
}


# -- POSITION axis: where the question sits ---------------------------------
#
# Not a renderer, because it changes the PROMPT rather than the context. Kept
# in the same file so the four axes are visible in one place.

#: The harness order: instructions, context, question.
POSITION_CONTEXT_FIRST = "{instructions}\n\nExcerpts:\n{context}\n\nQuestion: {question}\n"

#: Question before the evidence, so the model knows what it is looking for
#: while reading. Costs nothing and is the cheapest thing on this list.
POSITION_QUESTION_FIRST = "{instructions}\n\nQuestion: {question}\n\nExcerpts:\n{context}\n"

#: Both. Standard practice for long context and the reason is mechanical --
#: attention to the middle of a long input is weakest, and this puts the
#: question at both ends where it is strongest.
POSITION_SANDWICH = (
    "{instructions}\n\nQuestion: {question}\n\nExcerpts:\n{context}\n\n"
    "Question again: {question}\n"
)

POSITIONS: dict[str, str] = {
    "context_first": POSITION_CONTEXT_FIRST,
    "question_first": POSITION_QUESTION_FIRST,
    "sandwich": POSITION_SANDWICH,
}


__all__ = ["POSITIONS", "RENDERERS", "Record", "Renderer"]
