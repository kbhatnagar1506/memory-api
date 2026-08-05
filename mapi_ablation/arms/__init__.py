"""Arm registry: name -> builder.

Every arm receives the identical node set and edge set for a given
(seed, n_nodes). The ONLY thing that varies across arms is the encoding. That
property is what makes the comparison mean anything -- if an arm ever gets a
curated subset while another gets an unscoped dump, the experiment measures
selection rather than representation and the result is worthless.

`tests/test_arm_parity.py` parses each arm's payload back into node and edge
sets and asserts all four recover identical sets. Adding a fifth arm means
adding one file and one registry entry, and it must pass that test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..graph import Graph


@dataclass
class Payload:
    """What one arm hands to the model, plus what it claims to encode."""

    arm: str
    kind: str  # "text" | "image"
    #: One sentence naming the payload type. This is the ONLY part of the
    #: prompt wrapper permitted to differ between arms.
    descriptor: str
    text: str | None = None
    image_png: bytes | None = None
    media_type: str = "image/png"
    meta: dict = field(default_factory=dict)

    def payload_hash(self) -> str:
        blob = self.text.encode() if self.kind == "text" else (self.image_png or b"")
        return hashlib.sha256(blob).hexdigest()[:16]

    def size_note(self) -> str:
        if self.kind == "text":
            return f"{len(self.text or '')} chars"
        return f"{len(self.image_png or b'')} bytes PNG"


#: (id, label, type, domain, date)
NodeTuple = tuple[str, str, str, str, str]
#: (src, dst, rel), with `contradicts` normalized to sorted endpoints
EdgeTuple = tuple[str, str, str]


def normalize_edge(src: str, dst: str, rel: str) -> EdgeTuple:
    if rel == "contradicts":
        a, b = sorted((src, dst))
        return (a, b, rel)
    return (src, dst, rel)


class ArmModule(Protocol):
    NAME: str

    def build(self, g: Graph, **kw) -> Payload: ...
    def recover(self, p: Payload) -> tuple[set, set]: ...


from . import prose, json_arm, layout_text, canvas  # noqa: E402

ARMS: dict[str, ArmModule] = {
    m.NAME: m  # type: ignore[misc]
    for m in (prose, json_arm, layout_text, canvas)
}

TEXT_ARMS = tuple(n for n, m in ARMS.items() if n != "canvas")


def build(arm: str, g: Graph, **kw) -> Payload:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    return ARMS[arm].build(g, **kw)


def recover(p: Payload) -> tuple[set, set]:
    """Parse a payload back into (nodes, edges). Used by the parity test."""
    return ARMS[p.arm].recover(p)
