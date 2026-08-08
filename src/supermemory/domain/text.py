"""Text analysis for lexical retrieval.

This exists to make the in-memory backend agree with PostgreSQL. Postgres's
`english` text-search configuration removes stopwords and applies a Snowball
stemmer before indexing; a naive Python tokenizer does neither. The gap is not
cosmetic:

  * Without stopword removal, the query "how do we deploy" matches every
    document containing the word "we", and BM25 happily ranks an unrelated
    memory first. This was a real result from the demo corpus.
  * Without stemming, "deploy" does not match "Deploys", so the one document
    that answers the query is not retrieved at all.

The stemmer is a deliberately small suffix-stripper rather than full Porter.
It covers the inflections that matter for retrieval (plurals, -ing, -ed, -ly)
and stops there, because every additional rule is another way for two backends
to disagree. `test_text.py` pins the behaviour both must share.
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_POSSESSIVE_RE = re.compile(r"'s$|'$|^'")

#: Matches PostgreSQL's english stopword list closely enough for parity. Kept
#: short intentionally: an over-broad list silently deletes meaningful query
#: terms ("can" in "can bus", "it" in "it department").
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


@lru_cache(maxsize=8192)
def stem(word: str) -> str:
    """Light suffix stripping. Conservative: short words are left alone."""
    if len(word) <= 3:
        return word
    for suffix, minimum in (
        ("ational", 7),
        ("iveness", 8),
        ("fulness", 8),
        ("ousness", 8),
        ("ization", 8),
        ("ations", 7),
        ("ingly", 7),
        ("edly", 6),
        ("ement", 7),
        ("ness", 6),
        ("tion", 6),
        ("ment", 6),
        ("ies", 5),
        ("ing", 5),
        ("ers", 5),
        ("est", 5),
        ("ed", 4),
        ("es", 4),
        ("ly", 5),
        ("er", 5),
        ("s", 4),
    ):
        if len(word) >= minimum and word.endswith(suffix):
            trimmed = word[: -len(suffix)]
            if suffix == "ies":
                return trimmed + "y"
            if suffix == "es":
                # "boxes"/"dishes" drop the whole "es"; "upgrades" keeps the e.
                return (
                    trimmed if trimmed.endswith(("s", "x", "z", "ch", "sh")) else trimmed + "e"
                )
            if suffix in {"tion", "ation", "ational"}:
                return trimmed + "t" if suffix == "tion" else trimmed
            # Undo the doubled consonant in "stopping" -> "stop".
            if (
                suffix in {"ing", "ed"}
                and len(trimmed) > 2
                and trimmed[-1] == trimmed[-2]
                and trimmed[-1] not in "lsz"
            ):
                trimmed = trimmed[:-1]
            return trimmed
    return word


def tokenize(text: str) -> list[str]:
    """Casefold, split, and strip possessives and stray apostrophes.

    Without the trim, "what's" tokenizes to "what's", the stemmer strips the
    trailing s, and the term becomes "what'" -- which matches nothing.
    """
    out: list[str] = []
    for raw in _WORD_RE.findall(text.casefold()):
        cleaned = _POSSESSIVE_RE.sub("", raw).strip("'")
        if cleaned:
            out.append(cleaned)
    return out


def analyze(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Full analysis: tokenize, drop stopwords, stem.

    Returns an EMPTY list for input that is entirely stopwords. That is a real
    signal -- "how do we" carries no retrievable term -- and swallowing it with
    an automatic fallback would hide the distinction from callers who need it.
    The lexical index falls back deliberately; the reranker declines to reorder.
    """
    tokens = tokenize(text)
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def analyze_query(text: str) -> list[str]:
    """Analysis for a search query, with an explicit stopword fallback.

    Matching on stopwords is poor, but returning nothing at all for a query the
    user actually typed is worse, so the index keeps them rather than answering
    an all-stopword query with silence.
    """
    terms = analyze(text)
    return terms or analyze(text, keep_stopwords=True)


__all__ = ["STOPWORDS", "analyze", "analyze_query", "stem", "tokenize"]
