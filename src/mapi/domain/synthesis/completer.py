"""Completion provider for the derive path.

The product's read path is LLM-free by design — search never waits on a
model. Derivation is the one opt-in exception, and it goes through this
narrow seam: a `CompleteFn` is just `async (prompt) -> text`, so the service
never knows which vendor is behind it and tests inject a stub.

Native async client only. The blocking-client-under-`to_thread` pattern
deadlocked a 500-corpus benchmark run (a cancelled `wait_for` cannot cancel a
thread stuck on a socket read; every timeout permanently burned an executor
slot), so it is banned here even though this path is lower-volume.
"""

from __future__ import annotations

from ...config import Settings, SynthesisBackend
from ...core.errors import ConfigurationError
from ...core.logging import get_logger
from .derive import CompleteFn

log = get_logger(__name__)


def _gemini(
    settings: Settings, *, model: str, thinking_budget: int, max_output_tokens: int
) -> CompleteFn:
    """One Gemini completion callable, with both token budgets pinned.

    Both budgets are set EXPLICITLY, because leaving thinking unbounded is
    not a neutral default. On gemini-2.5-flash reasoning tokens are drawn
    from the same allowance as the answer, so an unbounded think against a
    1536 ceiling returned 196 characters of a JSON array cut mid-string --
    which parses to nothing and looks exactly like "this passage contained
    no facts". A silent, plausible zero is the worst failure shape
    available, so truncation is logged rather than swallowed.
    """
    if not settings.google_cloud_project:
        raise ConfigurationError("synthesis_backend=gemini requires google_cloud_project")
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        http_options=genai_types.HttpOptions(timeout=60_000),
    )

    async def complete(prompt: str) -> str:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_output_tokens,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )
        candidates = response.candidates or []
        reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
        if "MAX_TOKENS" in reason.upper():
            log.warning(
                "completion_truncated",
                model=model,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
                chars=len(response.text or ""),
            )
        return response.text or ""

    return complete


def build_completer(settings: Settings) -> CompleteFn | None:
    """The READ-path completer: derivation, run once per question."""
    if settings.synthesis_backend is SynthesisBackend.NONE:
        return None
    if settings.synthesis_backend is SynthesisBackend.GEMINI:
        return _gemini(
            settings,
            model=settings.synthesis_model,
            thinking_budget=settings.synthesis_thinking_budget,
            max_output_tokens=settings.synthesis_max_output_tokens,
        )
    raise ConfigurationError(f"unknown synthesis backend: {settings.synthesis_backend}")


def build_extractor(settings: Settings) -> CompleteFn | None:
    """The WRITE-path completer: extraction, run once per document.

    Separate from the read-path completer because the two jobs have nothing
    in common except the SDK. Derivation runs per question, reasons over
    several documents, and its quality shows up directly in an answer.
    Extraction runs on every write and is closer to transcription: read a
    passage, restate what it says, copy the quote that proves it.

    The constraint that actually binds here is THROUGHPUT, not quality per
    call. One LongMemEval ingest is hundreds of thousands of documents, so
    a per-document reasoning model turns a 90-minute ingest into a multi-day
    one. Same reason `extraction_thinking_budget` defaults low: this task
    has little to reason about, and we measured on the answer path that more
    thinking made things worse, not better.

    Split so both can be pointed at different models without one dragging
    the other.
    """
    if settings.synthesis_backend is SynthesisBackend.NONE:
        return None
    if settings.synthesis_backend is SynthesisBackend.GEMINI:
        return _gemini(
            settings,
            model=settings.extraction_model,
            thinking_budget=settings.extraction_thinking_budget,
            max_output_tokens=settings.extraction_max_output_tokens,
        )
    raise ConfigurationError(f"unknown synthesis backend: {settings.synthesis_backend}")


def build_understander(settings: Settings) -> CompleteFn | None:
    """The SEARCH-path completer: one sentence in, one word out.

    The narrowest of the three and the only one on the read path, so it is
    configured for latency above everything: the smallest model, thinking
    off, and an output allowance of a few tokens. Its caller wraps it in a
    timeout and a circuit breaker and falls back to regexes, which is what
    makes putting a model in front of search defensible at all.
    """
    if not settings.understand_queries:
        return None
    if settings.synthesis_backend is SynthesisBackend.NONE:
        return None
    if settings.synthesis_backend is SynthesisBackend.GEMINI:
        return _gemini(
            settings,
            model=settings.understanding_model,
            thinking_budget=settings.understanding_thinking_budget,
            max_output_tokens=settings.understanding_max_output_tokens,
        )
    raise ConfigurationError(f"unknown synthesis backend: {settings.synthesis_backend}")


__all__ = ["build_completer", "build_extractor", "build_understander"]
