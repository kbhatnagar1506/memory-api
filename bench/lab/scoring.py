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


#: Below this, a one-space variant is not generated. "onto" would otherwise
#: admit "on to", and two-letter needles would fragment into noise. Compounds
#: worth catching ("bodyweight", "slipons", "secondhand") are all longer.
_MIN_COMPOUND = 6


def _compound_forms(needle: str) -> list[str]:
    """The needle, plus the spellings English uses for the same compound.

    MEASURED, not anticipated. On the k=ALL synthesis run two correct answers
    were scored wrong for nothing but a space:

        spec "slipons"      answer "a pair of slip-ons"    -> slip ons
        spec "bodyweight"   answer "using your body weight"

    `_normalize` turns a hyphen into a space, so the hyphenated form and the
    closed form can never meet, and which of the three spellings a spec author
    types is arbitrary. Both directions are generated:

      * a closed needle gets ONE optional split point per position, so
        "bodyweight" reaches "body weight" -- and only that. Inserting a
        separator at every position at once would also match "b o d y w e i g
        h t", which is not a spelling of anything.
      * an open needle gets its spaces removed, so "second hand" reaches
        "secondhand".

    Word boundaries still apply to every variant, so this widens which
    SPELLINGS match and never which words do.
    """
    forms = [needle]
    if " " in needle:
        # Every part must be a word in its own right. Without this, "cat s"
        # closes to "cats" and matches a plain plural -- a one-letter token is
        # not half of a compound, it is a typo or a split that never happened.
        if all(len(part) >= 2 for part in needle.split()):
            forms.append(needle.replace(" ", ""))
    elif len(needle) >= _MIN_COMPOUND:
        forms.extend(needle[:i] + " " + needle[i:] for i in range(2, len(needle) - 1))
    return forms


def _right_edge(needle: str) -> str:
    """What may follow a needle without it being a different token.

    The general rule is "no letter or digit after", which is what stops "18"
    matching inside "180". Applied to a NUMBER it also stops "180" matching
    inside "180cm", and a live run scored two correct answers wrong for exactly
    that -- "The Okonjo fixture is 180cm tall" against a spec of `180`. Numbers
    arrive welded to their units constantly (40kg, 20mm, 85hr, 3200gbp) and a
    scorer that cannot read them is measuring its own lexer.

    So for an all-digit needle the guard narrows to "no DIGIT after". "18"
    still cannot match "180", because what follows is a digit; "180" can now
    match "180cm", because what follows is not. Non-numeric needles keep the
    strict rule, where a following letter genuinely does mean a different word.
    """
    return r"(?!\d)" if needle.isdigit() else r"(?![0-9a-z])"


def _contains(answer_norm: str, surface: str) -> bool:
    """Word-boundary containment of a normalized surface in a normalized answer.

    Substring alone would let "18" match inside "180" -- a wrong number scored
    as right, which is the one direction a scorer must never err in.
    """
    needle = _normalize(surface)
    if not needle:
        return False
    return any(
        re.search(rf"(?<![0-9a-z]){re.escape(form)}{_right_edge(form)}", answer_norm)
        for form in _compound_forms(needle)
    )


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
