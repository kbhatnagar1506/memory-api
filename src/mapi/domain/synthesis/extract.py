"""Write-time decomposition: turn a blob into the claims it contains.

Measured on 374 memories seeded from real LongMemEval chat histories: 100%
of them are exactly one chunk. The chunker targets 320 tokens and a chat
turn is ~50, so nothing splits, and one turn like

    "I'll definitely ask them about customized itineraries and Quetzal-
     focused tours. I've also been thinking about my photography gear and
     wanted to know if you have recommendations for lenses or camera
     settings for capturing birds in flight. I've had some experience with
     my Nikon camera..."

becomes ONE vector: the average of tour logistics, photography gear, and
owning a Nikon. Three consequences, all of which we measured before we
understood the cause:

  * RETRIEVAL. "What camera does he own" must match a vector that is mostly
    about itineraries. This is the signature of the largest surviving
    failure class -- 24 of 63 delivered-but-wrong answers were "right
    event, wrong attribute".
  * CONSOLIDATION. Cosine between two averaged blobs measures topic
    overlap, not claim identity, so "same subject (similarity 0.72)" fired
    on pairs where neither replaced the other -- 94 supersessions across
    six histories, median confidence 0.43.
  * THE GRAPH. Blobs share topics, not claims, so almost nothing connects
    and the picture is a field of isolated dots.

Splitting on sentences would not fix it: "I've had some experience with my
Nikon camera, but I'm not sure what would be the best setup" is one
sentence carrying one fact and one non-fact, and a pronoun in sentence
three is meaningless without sentence one. Deciding what a standalone claim
IS requires reading, which is why the systems that do this well spend a
model call on every write.

Two integrity rules, both enforced in code rather than asked for in the
prompt:

  * GROUNDED. A claim must carry a verbatim quote from its source or it is
    discarded. Extraction fabricates, and a fabricated memory is worse than
    a missing one because it is indistinguishable from a real one later.
  * ADDITIVE. Claims never replace the text they came from. If extraction
    misses the one attribute a question needed, the original is still
    stored and still retrievable; the worst case of this subsystem is the
    status quo, not lost evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ..text import STOPWORDS
from .derive import CompleteFn

#: Below this there is nothing to decompose, and a model call per one-line
#: write is pure cost. "more" and "please continue" are real turns in the
#: corpus; they hold no claim and must not become memories.
MIN_CHARS = 80

#: Jaccard over content tokens at or above which two claims are the same
#: claim. Deliberately below the 0.8 used elsewhere, because extracted claims
#: are short: "owns a Nikon camera" against "the user owns a Nikon camera" is
#: 0.75, one scaffolding word apart. The pairs that MUST stay separate differ
#: by a content word and score far lower -- road vs mountain bike is 0.50,
#: 27:12 vs 25:50 is 0.33 -- so the gap is wide, not marginal.
_DUPLICATE_JACCARD = 0.75

#: A blob yielding more claims than this is a transcript, not a statement.
#: The cap bounds cost and stops a runaway completion from writing hundreds
#: of memories against one document.
MAX_CLAIMS = 24

#: Every rule here is traceable to a measured failure, and the prompt is the
#: mechanism -- the retrieval unit is whatever this returns. Notes on the two
#: rules that are easiest to get backwards:
#:
#:   ATOMIC IS NOT MINIMAL. "Ran the charity 5K in 27:12 on 20 May 2023" is
#:   ONE claim. Splitting it into a run, a time and a date destroys the
#:   binding between them, which is precisely the failure being fixed: 24 of
#:   63 delivered-but-wrong answers had the right event and the wrong
#:   attribute. The unit is one assertion WITH its qualifiers attached.
#:
#:   PREFERENCES ARE FACTS. "Prefers window seats" is durable and is a whole
#:   benchmark category (single-session-preference). Only reactions to the
#:   assistant's suggestions get dropped. An earlier draft of this prompt said
#:   "skip opinions", which would have thrown that category away.
EXTRACT_PROMPT = """Extract the durable facts stated by {subject} in the passage below.

The passage was written on {as_of}.

Return a JSON array. Each element has exactly two keys:
  "fact"  - one self-contained statement about {subject}, in the third person
  "quote" - the shortest VERBATIM span of the passage that states it

RULES

1. ONE ASSERTION PER FACT, WITH ITS QUALIFIERS.
   Split a sentence that states two unrelated things. Do NOT split one thing
   into its parts: keep the time, date, place, quantity and name attached to
   the assertion they belong to.
   Right: "Ran the charity 5K in 27:12 on 20 May 2023."
   Wrong: "Ran a charity 5K." + "Had a time of 27:12." + "It was 20 May."

2. SELF-CONTAINED.
   Each fact is stored alone and will be read with nothing around it. Resolve
   every pronoun, "it", "there", "the same one", and every reference to
   earlier text.
   Right: "{subject} owns a Nikon camera."
   Wrong: "They have some experience with it."

3. RESOLVE TIME AGAINST {as_of}.
   Convert every relative date to an absolute one: "last Thursday", "two
   weeks ago", "next month". If the passage gives no timing, state none --
   do not guess one.

4. KEEP EVERY SPECIFIC.
   Numbers, units, times, dates, prices, brand names, model numbers, place
   names, people's names. The specific IS the fact; a fact stripped of it
   answers nothing.
   Right: "Beat their personal best with a time of 25:50."
   Wrong: "Improved their personal best."

5. ONLY WHAT THE PASSAGE STATES.
   No inference, no generalising, no combining, no summarising, no filling
   gaps from world knowledge. If it is not said, it is not a fact.

6. THE QUOTE MUST BE VERBATIM.
   Copy the span character for character from the passage. Do not paraphrase,
   correct or shorten mid-phrase. A fact whose quote is not in the passage is
   discarded.

WHAT COUNTS AS A DURABLE FACT
   Keep: events and when they happened; states and situations; possessions;
   relationships; plans and intentions; preferences, likes and dislikes;
   skills, jobs, health, habits; decisions made; things that changed,
   including things that stopped or are no longer true.
   Skip: questions and requests to the assistant; reactions to what the
   assistant suggested ("that's helpful", "I'll try that"); greetings and
   filler; anything about the assistant rather than {subject}.

A passage may contain no durable facts. Return [] -- that is a correct answer,
and inventing one to avoid an empty result is the worst thing you can do here.

Return the JSON array and nothing else.

PASSAGE:
{text}
"""


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic, self-contained fact, with the span that proves it."""

    fact: str
    quote: str


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


_TOKEN = re.compile(r"[a-z0-9]+")


def _content_tokens(fact: str) -> set[str]:
    """Tokens that distinguish one claim from another.

    Stopwords out before comparison, as in `derive`: "owns a road bike" and
    "owns a mountain bike" share owns/a/bike and would merge on raw Jaccard,
    with the scaffolding words outvoting the content words that veto it.
    """
    tokens = set(_TOKEN.findall(fact.casefold()))
    content = tokens - STOPWORDS
    return content or tokens


def _dedupe(claims: Sequence[Claim]) -> list[Claim]:
    """Drop near-identical restatements, keeping the first.

    A model asked to split a passage will often emit the same fact twice
    with different scaffolding. Two memories saying one thing is not
    harmless: it double-counts under any aggregate and gives one claim two
    votes in retrieval.
    """
    kept: list[Claim] = []
    seen: list[set[str]] = []
    for claim in claims:
        tokens = _content_tokens(claim.fact)
        if not tokens:
            continue
        duplicate = False
        for previous in seen:
            union = tokens | previous
            if union and len(tokens & previous) / len(union) >= _DUPLICATE_JACCARD:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(claim)
        seen.append(tokens)
    return kept


def parse_claims(raw: str, source: str) -> list[Claim]:
    """Parse a completion into grounded claims. Never raises.

    Grounding is a substring check against the source, whitespace-normalized
    and case-folded: strict enough that invented text cannot pass, loose
    enough that a reflowed quote still does. Same gate as the derive path,
    for the same reason -- this is the only thing standing between model
    output and something stored as a fact.
    """
    text = raw.strip()
    # Models fence JSON even when told not to.
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        rows = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(rows, list):
        return []

    haystack = _normalize(source)
    claims: list[Claim] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fact = str(row.get("fact", "")).strip()
        quote = str(row.get("quote", "")).strip()
        if not fact or not quote:
            continue
        if _normalize(quote) not in haystack:
            continue
        claims.append(Claim(fact=fact, quote=quote))
    return _dedupe(claims)[:MAX_CLAIMS]


#: Used when the caller does not name the person. A stable third-person
#: referent, because the facts are stored in the third person and "I" in a
#: memory read six months later refers to nobody.
DEFAULT_SUBJECT = "the user"


async def extract_claims(
    text: str,
    complete: CompleteFn,
    *,
    as_of: datetime | None = None,
    subject: str = DEFAULT_SUBJECT,
) -> list[Claim]:
    """Decompose `text` into grounded atomic claims.

    `as_of` is the passage's own date, and passing it is what lets "last
    Thursday" become a date. Without it the model has no anchor and either
    invents one or drops the timing -- both fatal for temporal questions, so
    the prompt says to state no timing rather than guess.

    Fail-open: any provider error, malformed completion or ungrounded row
    yields fewer claims or none, and the caller stores the original text
    exactly as it does today.
    """
    if len(text.strip()) < MIN_CHARS:
        return []
    when = (as_of or datetime.now(UTC)).strftime("%A %d %B %Y")
    try:
        raw = await complete(
            EXTRACT_PROMPT.format(text=text, as_of=when, subject=subject or DEFAULT_SUBJECT)
        )
    except Exception:
        return []
    return parse_claims(raw, text)


__all__ = [
    "DEFAULT_SUBJECT",
    "EXTRACT_PROMPT",
    "MAX_CLAIMS",
    "MIN_CHARS",
    "Claim",
    "extract_claims",
    "parse_claims",
]
