"""Grounded chat over a space's memories.

The whole point of a memory API is what an agent does with what it remembers,
and every part of that is here in one place: retrieve, ground, answer, cite.

Three properties separate this from putting a vector store behind a chat box:

  * IT CITES. Every memory in context is numbered, and the model is told to
    mark which numbers it used. A claim that names no source is a claim the
    memory did not support, and the caller can see that rather than infer it.
  * IT DECLINES. "I do not have that in memory" is a correct answer and the
    prompt says so explicitly. A chat that invents an answer from an empty
    result set is worse than one that returns nothing, because the failure is
    invisible.
  * IT REPORTS CONFIDENCE. Memories carry a `confidence` in metadata, and an
    unverified one is labelled as such in the context. A memory system that
    launders "someone told me this once" into a confident answer is actively
    harmful -- it will repeat a shaky number back in the voice of a checked
    one. So the model is instructed to carry the caveat through.

No vendor SDK here. The completion is a `CompleteFn`, same as everywhere else
in this package, so this is testable with a stub and the domain layer stays
free of imports it should not have.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ...core.logging import get_logger
from ..models import ScoredMemory
from .classify import QuestionKind, classify
from .derive import CompleteFn

log = get_logger(__name__)

#: How many turns of history travel with the question. Bounded because the
#: context window is finite and the memories are the point -- an unbounded
#: transcript would crowd out the retrieval it exists to support.
MAX_HISTORY_TURNS = 8
#: Characters of any one memory that reach the prompt.
MAX_MEMORY_CHARS = 700

CHAT_PROMPT = """\
You are answering from a memory store. The numbered memories below are \
everything you know; you have no other knowledge of this subject.

Rules:
1. Answer ONLY from the memories. If they do not contain the answer, say so \
plainly — "I don't have that in memory" — and stop. Do not fill the gap.
2. Cite the memories you used as [1], [2] inline. Every factual claim needs \
one.
3. A memory marked UNVERIFIED is not established fact. If you use one, say it \
is unverified and why it matters, in the same sentence.
4. Where memories disagree, say so and give both. Do not silently pick one.
5. Be brief and concrete. No preamble, no restating the question, no offers \
of further help.

{history}Memories:
{memories}

Question: {question}
Answer:"""

#: The advice variant, and the measurement behind it.
#:
#: `CHAT_PROMPT`'s rule 1 -- decline when the memories do not contain the answer
#: -- is CORRECT for facts and is the measured failure mode for advice. An
#: advice request has no stored answer by construction; its whole point is to
#: APPLY stored preferences. Measured on the preference lab (12 questions,
#: identical evidence): the fact framing scored 7/12 with three refusals on
#: questions the advice framing answered correctly at 10/12.
#:
#: The two-step shape is the lab's `grounded` arm, which tied plain advice on
#: accuracy and won on legibility: where no preference bore on the request, the
#: plain arm confidently recommended "a weekday morning" -- the exact violation
#: of a stored never-before-10am rule whose memory had not been retrieved --
#: while the grounded arm wrote "no stored preference applies" and labelled its
#: suggestion a guess. Same score, different failure: a wrong answer wearing
#: confidence versus a system reporting its own evidence gap. Quoting the
#: preference FIRST also binds the recommendation to a citation, which is the
#: same mechanism that grounds the fact path.
ADVICE_CHAT_PROMPT = """\
You are advising this user from a memory store. The numbered memories below \
are what you know about them; they will NOT contain a ready-made answer -- \
they contain what this user likes, avoids, owns and does.

Work in two steps, both in your reply:
1. Name the preference(s) that bear on this request, quoting the memory and \
citing it as [1], [2] inline. If a preference CHANGED over time, use the \
latest. If NO memory bears on the request, say "no stored preference applies \
here" instead.
2. Make one concrete recommendation that follows from exactly the preferences \
you cited -- never from general taste. If none applied, still recommend, and \
say plainly that it is a guess.

A memory marked UNVERIFIED is not established fact; say so if you lean on it. \
Be brief and concrete.

{history}Memories:
{memories}

Request: {question}
Answer:"""


@dataclass(slots=True)
class Turn:
    """One exchange. `role` is "user" or "assistant"."""

    role: str
    content: str


@dataclass(slots=True)
class ChatAnswer:
    reply: str
    #: Memory ids the model marked as used, in citation order. Empty when it
    #: cited nothing, which is the shape of an honest refusal.
    cited: list[str] = field(default_factory=list)
    #: Every memory that reached the prompt, cited or not -- so a caller can
    #: see what the model was given and judge the ones it ignored.
    considered: list[str] = field(default_factory=list)
    #: True when the answer leaned on at least one unverified memory.
    used_unverified: bool = False


def _is_unverified(memory_metadata: dict[str, object] | None) -> bool:
    value = (memory_metadata or {}).get("confidence")
    return isinstance(value, str) and value.lower() in {"unverified", "low", "unsure"}


def format_memories(hits: Sequence[ScoredMemory]) -> str:
    """Number the memories and label the ones that are not established.

    Numbered rather than keyed by id: a 30-character ULID in a citation is
    thirty chances to transpose a character into an id that exists, and a
    mistyped id attaches a real claim to the wrong memory.

    The date is EVENT time, not write time. "Krishna joined Reakon" dated May
    2026 tells the model when the fact became true; the date it was typed in
    tells it nothing it can reason with.
    """
    lines = []
    for i, hit in enumerate(hits, start=1):
        memory = hit.memory
        text = (memory.content or "").strip()[:MAX_MEMORY_CHARS]
        stamp = ""
        if isinstance(memory.occurred_at, datetime):
            stamp = f" ({memory.occurred_at:%b %Y})"
        flag = " [UNVERIFIED]" if _is_unverified(memory.metadata) else ""
        lines.append(f"{i}.{stamp}{flag} {text}")
    return "\n".join(lines)


def format_history(turns: Sequence[Turn]) -> str:
    if not turns:
        return ""
    recent = list(turns)[-MAX_HISTORY_TURNS:]
    body = "\n".join(f"{t.role.capitalize()}: {t.content.strip()}" for t in recent)
    return f"Conversation so far:\n{body}\n\n"


#: Three digits, because a coverage question can now return more than 99.
#:
#: This was `\d{1,2}`, which was sufficient while the coverage window was capped
#: at 32 by a hydration-pool bug. Fixing that cap raised the ceiling to
#: `_MAX_COVERAGE_LIMIT` (200) and made memories 100 and beyond UNCITABLE: the
#: model would write `[137]`, the regex would not match, and the claim would be
#: reported as uncited -- indistinguishable from a claim the memories did not
#: support, which is the exact signal `cited` exists to carry.
#:
#: Bounded at three rather than unbounded: `\d+` would match a year in `[2026]`
#: or a bracketed figure in the memory text quoted back, and out-of-range
#: indices are dropped below anyway.
_CITATION = re.compile(r"\[(\d{1,3})\]")


def parse_citations(reply: str, hits: Sequence[ScoredMemory]) -> list[str]:
    """Memory ids for the [n] markers in the reply, in order, deduplicated.

    Out-of-range numbers are dropped rather than clamped: a citation to [9]
    when eight memories were supplied is the model inventing provenance, and
    silently pointing it at the eighth memory would manufacture exactly the
    audit trail this function exists to provide.
    """
    seen: set[int] = set()
    ids: list[str] = []
    for raw in _CITATION.findall(reply):
        index = int(raw)
        if not 1 <= index <= len(hits) or index in seen:
            continue
        seen.add(index)
        ids.append(hits[index - 1].memory.id)
    return ids


async def answer(
    question: str,
    hits: Sequence[ScoredMemory],
    complete: CompleteFn,
    *,
    history: Sequence[Turn] = (),
) -> ChatAnswer:
    """Answer `question` from `hits` alone.

    With no hits the model is never called: there is nothing to ground an
    answer in, and asking anyway invites exactly the invention rule 1 forbids.
    That holds for advice too -- with an EMPTY store there are no preferences to
    apply, and a recommendation from nothing is a recommendation from the
    model's own priors wearing the product's voice.

    Routing is by the question's SHAPE, via the same classifier the derive path
    and the benchmark use -- no dataset label, no caller flag. A fact question
    gets the decline-when-absent contract; an advice request gets the
    grounded-advice contract, because the decline rule is the measured failure
    mode there (7/12 -> 10/12 on the preference lab, three refusals converted).
    """
    considered = [h.memory.id for h in hits]
    if not hits:
        return ChatAnswer(
            reply="I don't have anything in memory about that.",
            considered=considered,
        )

    template = ADVICE_CHAT_PROMPT if classify(question) is QuestionKind.ADVICE else CHAT_PROMPT
    prompt = template.format(
        history=format_history(history),
        memories=format_memories(hits),
        question=question.strip(),
    )
    try:
        raw = await complete(prompt)
    except Exception as exc:
        log.warning("chat_completion_failed", error=str(exc)[:200])
        raise

    reply = (raw or "").strip()
    cited = parse_citations(reply, hits)
    cited_set = set(cited)
    used_unverified = any(
        _is_unverified(h.memory.metadata) for h in hits if h.memory.id in cited_set
    )
    return ChatAnswer(
        reply=reply,
        cited=cited,
        considered=considered,
        used_unverified=used_unverified,
    )


__all__ = [
    "ADVICE_CHAT_PROMPT",
    "CHAT_PROMPT",
    "ChatAnswer",
    "Turn",
    "answer",
    "format_history",
    "format_memories",
    "parse_citations",
]
