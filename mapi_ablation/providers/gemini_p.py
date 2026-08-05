"""Gemini adapter. Vertex AI via ADC by default, or the Developer API with a key."""

from __future__ import annotations

import os

from .base import ProviderError, Response, resolve_model, timed

NAME = "gemini"


class GeminiAdapter:
    name = NAME

    def __init__(self, model_candidates: list[str], temperature: float | None = 0.0):
        from google import genai

        self.temperature = temperature
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

        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            **({"temperature": self.temperature} if self.temperature is not None else {}),
            **({"system_instruction": system} if system else {}),
        )

        resp = None
        error = None
        with timed() as t:
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=cfg,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            return Response("", None, None, t.ms, error=error)

        usage = resp.usage_metadata
        return Response(
            text=(resp.text or ""),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=t.ms,
            raw={"finish_reason": str(
                resp.candidates[0].finish_reason if resp.candidates else None
            )},
        )
