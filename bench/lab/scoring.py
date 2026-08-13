"""Scoring that cannot invent failures.

The format experiment's headline "finding" was four questions failing in every
format. Three of the four were this module's predecessor:

    expected 'yes, 180 vs 210'    model said 'Yes'          -> scored WRONG
    expected 'aisle, no red-eye'  model said 'aisle seat'   -> scored WRONG
    expected 'Datadog 890'        model said 'Datadog'      -> scored WRONG

The old scorer split the expected string on spaces and demanded every token
appear. So `"aisle,"` -- with its comma -- could never match, and an answer we
had explicitly prompted to be terse ("answer with the specific value, in as few
words as possible") was punished for being terse. A scorer that disagrees with
the prompt is measuring the disagreement, not the system.

The repair is structural, not a tweak:

  * REQUIRED versus SUPPORTING. "Is the hotel cheaper?" is answered by "yes";
    "180 vs 210" is evidence a fuller answer would cite. Only `must` groups
    decide correctness; `bonus` tokens are reported as completeness, separately,
    because "right" and "thorough" are different measurements.
  * ALTERNATIVES. "red-eye", "red eye" and "redeye" are one answer. Each `must`
    group is a set of acceptable surfaces, any one of which satisfies it.
  * NORMALIZED MATCHING. Casefold, strip punctuation, match on word boundaries
    -- so "aisle." matches "aisle" and "1,500" matches "1500", but "18" does not
    match inside "180".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    """Casefold and strip punctuation/symbols, keeping word boundaries.

    Commas INSIDE numbers are removed rather than spaced: "$1,500" must
    normalize to "1500", not to "1 500" -- the general punctuation pass would
    split the number and "1500" would never match it. Decimal points go through
    the general pass, which is consistent on both sides: "4.2" in an expectation
    and "4.2" in an answer both become "4 2" and still match each other.
    """
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    out = []
    for ch in text.casefold():
        if unicodedata.category(ch).startswith(("P", "S")):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _contains(answer_norm: str, surface: str) -> bool:
    """Word-boundary containment of a normalized surface in a normalized answer.

    Substring alone would let "18" match inside "180" -- a wrong number scored
    as right, which is the one direction a scorer must never err in.
    """
    needle = _normalize(surface)
    if not needle:
        return False
    return re.search(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])", answer_norm) is not None


@dataclass(frozen=True, slots=True)
class Expected:
    """What a correct answer must contain, and what a thorough one would.

    `must`: every group must be satisfied; a group is satisfied by ANY of its
    surfaces. `bonus`: reported as completeness, never required.
    """

    must: tuple[tuple[str, ...], ...]
    bonus: tuple[str, ...] = ()

    @classmethod
    def parse(cls, spec: str) -> Expected:
        """The compact authoring form used in corpus files.

        Groups separated by `;`, alternatives within a group by `|`, bonus
        tokens after `+`:

            "yes + 180 210"          -> must [("yes",)], bonus ("180","210")
            "aisle; red-eye|red eye" -> must [("aisle",), ("red-eye","red eye")]
        """
        spec, _, bonus_part = spec.partition("+")
        must = tuple(
            tuple(alt.strip() for alt in group.split("|") if alt.strip())
            for group in spec.split(";")
            if group.strip()
        )
        bonus = tuple(t for t in bonus_part.split() if t)
        if not must:
            raise ValueError(f"expected spec has no required groups: {spec!r}")
        return cls(must=must, bonus=bonus)


@dataclass(frozen=True, slots=True)
class Score:
    """Correctness and completeness, separately, with what was missing."""

    correct: bool
    completeness: float  #: bonus tokens present / bonus tokens defined; 1.0 if none
    missing: tuple[str, ...] = field(default=())

    @property
    def complete(self) -> bool:
        return self.correct and self.completeness == 1.0


def score(answer: str, expected: Expected) -> Score:
    """Grade an answer. Refusals and errors are simply answers with no tokens."""
    answer_norm = _normalize(answer or "")
    missing: list[str] = []
    for group in expected.must:
        if not any(_contains(answer_norm, surface) for surface in group):
            missing.append("|".join(group))
    correct = not missing

    if expected.bonus:
        present = sum(1 for token in expected.bonus if _contains(answer_norm, token))
        completeness = present / len(expected.bonus)
    else:
        completeness = 1.0

    return Score(correct=correct, completeness=completeness, missing=tuple(missing))


__all__ = ["Expected", "Score", "score"]
