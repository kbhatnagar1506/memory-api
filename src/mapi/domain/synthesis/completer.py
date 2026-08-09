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
from .derive import CompleteFn


def build_completer(settings: Settings) -> CompleteFn | None:
    """The configured completion callable, or None when synthesis is off."""
    if settings.synthesis_backend is SynthesisBackend.NONE:
        return None
    if settings.synthesis_backend is SynthesisBackend.GEMINI:
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
        model = settings.synthesis_model

        async def complete(prompt: str) -> str:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    # Headroom for thinking models: reasoning tokens share the
                    # output budget, and a too-tight cap truncates mid-answer.
                    max_output_tokens=1536,
                ),
            )
            return response.text or ""

        return complete
    raise ConfigurationError(f"unknown synthesis backend: {settings.synthesis_backend}")


__all__ = ["build_completer"]
