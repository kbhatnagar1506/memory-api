"""The single shared instruction wrapper.

There is exactly ONE wrapper and all four arms use it. Every byte is identical
across arms except:

  * the payload block itself, and
  * the one sentence that names the payload type (Payload.descriptor).

`tests/test_arm_parity.py` asserts that property by diffing the wrappers with
the payload and descriptor removed. If a change makes the wrappers differ in any
other way, the arms are no longer comparable and the experiment is void.

Two prompt conditions run as a factor:

  cold   -- nothing about the grammar beyond what the payload itself carries.
            This is the condition that matters for the "open spec, any model can
            read it" claim.
  primed -- three sentences describing the grammar. The cold/primed gap measures
            how much adoption gravity the spec would actually need.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arms import Payload

PROMPT_VERSION = "v1"

_INTRO = (
    "You are reading a memory record of a software project. It contains dated "
    "facts about the project and typed relationships between them."
)

#: Three sentences, used only in the `primed` condition.
_PRIMER = (
    "If the record is an image, read it as follows. The horizontal axis is "
    "time, with older items on the left and newer items on the right, and each "
    "horizontal band is one domain, labelled at the left edge. Each node box "
    "shows its ID in large text, its date, and a short label, and the weight of "
    "the box border encodes the node's type. A solid line with an arrowhead "
    "means the source caused the target, a dashed line with a bar at each end "
    "means the two nodes contradict each other, and a solid line with a double "
    "chevron means the node at the chevron end supersedes the node at the other "
    "end."
)

_ANSWER_RULE = (
    "Answer with a single node ID and nothing else. A node ID looks like N07. "
    "Do not explain your reasoning, do not restate the question, and do not add "
    "punctuation."
)

CONDITIONS = ("cold", "primed")


@dataclass
class BuiltPrompt:
    system: str
    text: str
    image_png: bytes | None
    condition: str
    arm: str
    prompt_version: str = PROMPT_VERSION

    @property
    def has_image(self) -> bool:
        return self.image_png is not None


def _wrapper(descriptor: str, condition: str, question: str,
             payload_text: str | None) -> str:
    """Assemble the wrapper. Only `descriptor` and the payload block vary."""
    parts = [_INTRO, "", f"The record below is {descriptor}."]
    if condition == "primed":
        parts += ["", _PRIMER]
    parts += ["", "--- BEGIN RECORD ---"]
    parts += [payload_text if payload_text is not None else "[image above]"]
    parts += ["--- END RECORD ---", "", f"Question: {question}", "", _ANSWER_RULE]
    return "\n".join(parts)


def build_prompt(payload: Payload, question: str, condition: str) -> BuiltPrompt:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return BuiltPrompt(
        system=_INTRO,
        text=_wrapper(
            payload.descriptor,
            condition,
            question,
            payload.text if payload.kind == "text" else None,
        ),
        image_png=payload.image_png if payload.kind == "image" else None,
        condition=condition,
        arm=payload.arm,
    )


def wrapper_skeleton(condition: str, question: str) -> str:
    """The wrapper with descriptor and payload removed.

    Used by the parity test: this string must be identical for all four arms.
    """
    return _wrapper("<DESCRIPTOR>", condition, question, "<PAYLOAD>")
