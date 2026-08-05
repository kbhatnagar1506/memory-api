"""Provider adapter interface.

Every adapter returns the same `Response`, and `input_tokens` MUST come from the
provider's own usage field. Never estimate it -- the legibility measurement
infers effective image scaling from reported token counts, so an estimated token
count would silently turn a measurement into a guess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Response:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    raw: Any = field(repr=False, default=None)
    #: Set when the call failed. `text` is then "" and the outcome is recorded
    #: as an error rather than a wrong answer.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ProviderAdapter(Protocol):
    name: str
    model: str
    temperature: float | None

    def complete(self, text: str, image_png: bytes | None = None,
                 system: str | None = None, max_tokens: int = 16) -> Response: ...

    def list_models(self) -> list[str]: ...


class ProviderError(RuntimeError):
    pass


def timed() -> "Timer":
    return Timer()


class Timer:
    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000

    ms: float = 0.0


def resolve_model(candidates: list[str], available: list[str], provider: str) -> str:
    """Pick the first candidate prefix that matches a currently-available model.

    Model IDs are resolved against the provider's live model list rather than
    hardcoded, and the resolved ID is written into the run manifest. If nothing
    matches we raise: running against a silently-substituted model would make
    the results uninterpretable.
    """
    for cand in candidates:
        exact = [m for m in available if m == cand or m.endswith("/" + cand)]
        if exact:
            return exact[0].split("/")[-1]
    for cand in candidates:
        pref = sorted(
            m for m in available if m.split("/")[-1].startswith(cand)
        )
        if pref:
            return pref[-1].split("/")[-1]
    raise ProviderError(
        f"{provider}: none of {candidates} matched the {len(available)} available "
        f"models. Available sample: {sorted(available)[:12]}"
    )
