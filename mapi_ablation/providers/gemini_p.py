"""Gemini adapter. Vertex AI via ADC by default, or the Developer API with a key."""

from __future__ import annotations

import os

from .base import ProviderError, Response, resolve_model, timed

NAME = "gemini"


class GeminiAdapter:
    name = NAME

    def __init__(self, model_candidates: list[str], temperature: float | None = 0.0,
                 thinking_budget: int | None = 128):
        from google import genai

        self.temperature = temperature
        # Gemini 2.5+ are thinking models: reasoning tokens are drawn from the
        # SAME output budget as the answer. At max_tokens=24, gemini-2.5-pro
        # spent 19 tokens thinking and returned a truncated "N", which scores
        # `unparseable` -- a harness artefact that would have looked exactly
        # like an unreadable grammar. `complete()` therefore reserves room for
        # thinking on top of the caller's answer budget.
        self.thinking_budget = thinking_budget
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.backend = "developer-api"
        else:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ProviderError(
                    "Set GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT to use Vertex AI "
                    "with Application Default Credentials "
                    "(gcloud auth application-default login)."
                )
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
            self.backend = "vertex-adc"
        self.model = resolve_model(model_candidates, self.list_models(), NAME)
        self.model_resolution = f"resolved against live models.list() ({self.backend})"

    def list_models(self) -> list[str]:
        try:
            return [m.name for m in self.client.models.list() if m.name]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"gemini: could not list models: {exc}") from exc

    def complete(self, text: str, image_png: bytes | None = None,
                 system: str | None = None, max_tokens: int = 16) -> Response:
        from google.genai import types

        parts: list = []
        if image_png is not None:
            parts.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
        parts.append(types.Part.from_text(text=text))

        base: dict = {
            **({"temperature": self.temperature} if self.temperature is not None else {}),
            **({"system_instruction": system} if system else {}),
        }
        budget = self.thinking_budget or 0
        # Answer budget plus headroom for reasoning tokens, which share it.
        out_cap = max_tokens + budget + 64

        def make_cfg(with_thinking: bool):
            kw = dict(base, max_output_tokens=out_cap)
            if with_thinking and self.thinking_budget is not None:
                kw["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            return types.GenerateContentConfig(**kw)

        resp = None
        error = None
        with timed() as t:
            for with_thinking in (True, False):
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=[types.Content(role="user", parts=parts)],
                        config=make_cfg(with_thinking),
                    )
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    # Some models reject an explicit thinking budget outright;
                    # fall back once rather than losing the cell.
                    if "thinking" not in str(exc).lower():
                        break
        if error is not None:
            return Response("", None, None, t.ms, error=error)

        usage = resp.usage_metadata
        finish = str(resp.candidates[0].finish_reason if resp.candidates else None)
        text = resp.text or ""
        # A truncated answer is a harness failure, not a model error. Surface it
        # as one rather than letting it score as `unparseable`.
        if "MAX_TOKENS" in finish:
            return Response(
                text, getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None), t.ms,
                error=f"output truncated at max_output_tokens={out_cap} "
                      f"(thoughts={getattr(usage, 'thoughts_token_count', None)})",
            )
        return Response(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=t.ms,
            raw={
                "finish_reason": finish,
                "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
            },
        )
