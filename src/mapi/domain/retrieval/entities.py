"""Entity extraction for cross-session retrieval bridging.

Measured motivation, from our own LoCoMo run: on multi-hop questions the
supporting turns sit a **median 204 turns apart**, and **97% of them are in
different sessions**. Only 2.2% fall inside a radius-2 window. Widening the
window or inflating top-k cannot reach that evidence — a question and its
second supporting fact often share no content words at all, only a subject.

What they do share is an *entity*. "What degree did Caroline get?" and
"Caroline enrolled at State" are 204 turns apart and lexically almost
disjoint, but both name Caroline. So: retrieve seeds, pull the salient
entities out of them, and look those entities up directly. One indexed lookup
bridges a gap that similarity search cannot.

MEASURED RESULT: this does not work, on any corpus tested. Keep it only as a
documented negative.

  * LoCoMo: 0.10 usable entities per turn, 91% of turns have none, 63 distinct
    entities in a 5,882-turn corpus. Nothing to bridge with. 0 results
    attributed to bridging.
  * LongMemEval: ~50 entities per session, so density was not the blocker --
    and it still produced IDENTICAL full recall (0.888) at 2.6x the latency
    (6,666ms vs 2,537ms) on n=500.

The dispersion diagnosis that motivated it was correct (97% of multi-hop
evidence is in another session, median 204 turns away). The remedy was not:
entity co-occurrence is too weak a bridge, and where entities are dense they
are generic ("use", "consider", "keep"). Disabled by default; do not enable
without re-measuring on the target corpus.

No LLM. Extraction is a heuristic over capitalisation and shape, which keeps
the write path lossless and free — the property that distinguishes us from
extraction-based memory systems, and which the independent cost study
(arXiv 2603.04814) found beats them on recall anyway.

The heuristic is deliberately shallow. A real NER model would find more, but it
would also add a torch dependency to the read path and a second thing to keep
in sync between backends. If the measured gain justifies it, upgrading the
extractor is a local change behind this same interface.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from ..text import STOPWORDS

#: Capitalised runs: "Caroline", "State University", "New York City".
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
#: ALLCAPS acronyms and codes: "NASA", "TS2345". Length-bounded to skip shouting.
_ACRONYM = re.compile(r"\b([A-Z]{2,6}\d{0,5})\b")
#: Years and numeric identifiers, which anchor temporal and order questions.
_NUMERIC = re.compile(r"\b(\d{4}|\d{3,})\b")
#: Identifiers that name a thing without being capitalised.
#:
#: The capitalisation patterns above encode an assumption -- that a name is
#: a capitalised word -- which holds for people and places and fails for
#: everything a technical or commercial corpus is made of. `postgres`, `k8s`,
#: `SKU-4471`, `eBay`, `pgvector`, `gpt-4o` and `v2.1.3` are all names, and
#: not one of them matches `[A-Z][a-z]{2,}`. These three patterns cover the
#: shapes that carry identity without carrying a capital letter:
#:
#:   * internal capitals   camelCase, eBay, PyPI, iPhone
#:   * digit-bearing words k8s, gpt-4o, s3, utf8
#:   * punctuated compounds  SKU-4471, text-embedding-004, v2.1.3
_CAMEL = re.compile(r"\b([a-z]+[A-Z][A-Za-z]*|[A-Z][a-z]*[A-Z][A-Za-z]*)\b")
_ALNUM_ID = re.compile(r"\b([a-z]+\d+[a-z\d]*|\d+[a-z]+[a-z\d]*)\b", re.IGNORECASE)
_PUNCT_ID = re.compile(r"\b([A-Za-z][A-Za-z\d]*(?:[-_.][A-Za-z\d]+){1,4})\b")

#: There is no blocklist.
#:
#: There used to be one: 79 hand-picked words that "look like names but are
#: not", which is a category that cannot be enumerated. It contained `may`,
#: `march`, `monday`, `user` and `assistant`, so a memory about the company
#: Monday.com, a product called March, or a person named May was invisible to
#: this stage -- and every other word nobody thought of stayed invisible too.
#:
#: Two signals already present do the job without a list. `STOPWORDS` covers
#: the grammatical openers ("The", "This", "And"), and `_is_sentence_initial`
#: covers the rest by position: "Wow" and "Cool" are only ever capitalised
#: because they start a sentence, and a real name eventually appears in the
#: middle of one. That positional rule is why the blocklist was mostly
#: redundant, and deleting it costs nothing the mid-sentence check was not
#: already catching.
#: A speaker prefix ("Caroline: ...") names the speaker, not a topic entity —
#: but in a two-person dialogue every turn carries one, so treating them as
#: salient would make every turn "match" every other. Stripped before extraction.
_SPEAKER_PREFIX = re.compile(r"^\s*[A-Z][\w .'-]{0,40}:\s*", re.MULTILINE)


def _is_sentence_initial(body: str, start: int) -> bool:
    """True if the match at `start` is the first word of a sentence.

    English capitalises sentence openers, so "Wow, great" and "Cool idea" look
    exactly like proper nouns to a shape-based matcher. Measured consequence:
    the first version of this extractor returned ['wow', 'melanie', 'cool',
    'mel'] as the salient entities of a dialogue, and bridging contributed
    zero results. Position is the signal that separates the two cases.
    """
    prefix = body[:start].rstrip()
    return not prefix or prefix[-1] in ".!?;:\n"


def scan(text: str) -> tuple[Counter[str], set[str]]:
    """Raw candidates and the subset seen mid-sentence.

    Split out from `extract_entities` so `salient_entities` can pool the
    mid-sentence evidence ACROSS seeds. A name that only ever opens a sentence
    in one short turn ("Caroline enrolled at State") is indistinguishable from
    "Wow" in that turn alone, but is confirmed the moment any other seed uses
    it mid-sentence.
    """
    counts: Counter[str] = Counter()
    mid_sentence: set[str] = set()
    if not text or not text.strip():
        return counts, mid_sentence
    body = _SPEAKER_PREFIX.sub("", text)

    for pattern in (_PROPER, _ACRONYM):
        for match in pattern.finditer(body):
            candidate = match.group(1).strip()
            if len(candidate) < 3:
                continue
            head = candidate.split()[0].casefold()
            if head in STOPWORDS:
                continue
            key = candidate.casefold()
            counts[key] += 1
            if not _is_sentence_initial(body, match.start(1)):
                mid_sentence.add(key)

    for pattern in (_CAMEL, _ALNUM_ID, _PUNCT_ID):
        for match in pattern.finditer(body):
            candidate = match.group(1).strip()
            key = candidate.casefold()
            if len(candidate) < 3 or key in STOPWORDS:
                continue
            counts[key] += 1
            # Shape, not position, is the evidence here: `gpt-4o` is an
            # identifier wherever it appears in the sentence, so these are
            # confirmed on sight rather than needing a mid-sentence sighting.
            mid_sentence.add(key)

    for match in _NUMERIC.findall(body):
        # A bare small integer is noise; a year or a long identifier is not.
        if len(match) >= 3:
            counts[match] += 1
            mid_sentence.add(match)
    return counts, mid_sentence


def extract_entities(text: str, *, max_entities: int = 12) -> list[str]:
    """Salient entity strings from ONE text, most frequent first.

    A capitalised token counts only if it appears mid-sentence at least once
    here, where capitalisation carries information — the difference between
    "Wow" and "Melanie". Callers with several related texts should prefer
    `salient_entities`, which pools that evidence and so recovers names that
    happen to open every sentence they appear in.

    Casefolded on return so callers can compare against analyzed index terms
    without worrying about capitalisation.
    """
    counts, mid_sentence = scan(text)
    found = Counter({k: v for k, v in counts.items() if k in mid_sentence})
    return [entity for entity, _ in found.most_common(max_entities)]


def salient_entities(
    texts: Sequence[str],
    *,
    query: str = "",
    max_entities: int = 8,
    #: An entity in half the seeds is background, not signal. Measured: at 0.9
    #: the speakers' own names survived in dialogue and expanding on them
    #: returned the whole conversation.
    drop_ubiquitous_ratio: float = 0.5,
    speakers: Sequence[str] = (),
) -> list[str]:
    """Entities worth expanding on, gathered across seed documents.

    Two filters matter:

    * Entities already in the query are dropped — the first-stage retriever
      has searched for them, so expanding on them re-fetches the same
      neighbourhood and wastes a lookup.
    * Entities appearing in nearly every seed are dropped. In a two-person
      dialogue both names appear constantly; expanding on them returns the
      whole corpus and drowns the signal. `drop_ubiquitous_ratio` is the
      share of seeds above which an entity is considered background.
    * `speakers` are dropped outright where the caller knows them.
    """
    if not texts:
        return []
    query_terms = set(extract_entities(query)) if query else set()
    # Speaker names are the strongest false positive in dialogue: they appear
    # everywhere, so they match everything and bridge nothing.
    banned = {s.casefold() for s in speakers}

    # Two passes: pool mid-sentence confirmations across every seed, then count
    # occurrences of whatever survived.
    tally: Counter[str] = Counter()
    confirmed: set[str] = set()
    per_seed: list[Counter[str]] = []
    for text in texts:
        counts, mid_sentence = scan(text)
        per_seed.append(counts)
        confirmed |= mid_sentence
    for counts in per_seed:
        tally.update({k for k in counts if k in confirmed})

    # A zero budget means zero, and it did not.
    #
    # The cap was checked AFTER the append, so `max_entities=0` returned one
    # entity: the first candidate was added, `1 >= 0` broke the loop, and the
    # caller got exactly what it had asked not to get. Harmless while the budget
    # was unreachable from the API; now that `entity_budget` is settable, a
    # caller disabling bridging by zeroing it would still have paid for one
    # indexed lookup per search.
    if max_entities <= 0:
        return []

    threshold = max(2, int(len(texts) * drop_ubiquitous_ratio))
    out: list[str] = []
    for entity, count in tally.most_common():
        if entity in query_terms or entity in banned:
            continue
        if len(texts) > 2 and count >= threshold:
            continue
        out.append(entity)
        if len(out) >= max_entities:
            break
    return out


def coverage(entities: Iterable[str], text: str) -> float:
    """Share of `entities` present in `text`. Used for explain traces."""
    items = list(entities)
    if not items:
        return 0.0
    lowered = text.casefold()
    return sum(1 for e in items if e in lowered) / len(items)


__all__ = ["coverage", "extract_entities", "salient_entities", "scan"]
