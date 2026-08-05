"""Anthropic adapter. Direct API key, or Vertex AI via ADC."""

from __future__ import annotations

import base64
import os

from .base import ProviderError, Response, resolve_model, timed

NAME = "anthropic"


class AnthropicAdapter:
    name = NAME

    def __init__(self, model_candidates: list[str], temperature: float | None = 0.0,
                 backend: str | None = None):
        self.temperature = temperature
        self.backend = backend or os.getenv("MAPI_ANTHROPIC_BACKEND", "api")
        if self.backend == "vertex":
            from anthropic import AnthropicVertex

            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            region = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")
            if not project:
                raise ProviderError("GOOGLE_CLOUD_PROJECT required for vertex backend")
            self.client = AnthropicVertex(project_id=project, region=region)
            # Vertex exposes no Claude model-list endpoint, so the candidate is
            # validated by the first real call rather than up front. The manifest
            # records that distinction instead of implying it was verified.
            self.model = model_candidates[0]
            self.model_resolution = "vertex: from config, not list-verified"
        else:
            from anthropic import Anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ProviderError(
                    "ANTHROPIC_API_KEY is not set. Either export it, or set "
                    "MAPI_ANTHROPIC_BACKEND=vertex to use Vertex AI with ADC."
                )
            self.client = Anthropic()
            self.model = resolve_model(model_candidates, self.list_models(), NAME)
            self.model_resolution = "resolved against live models.list()"

    def list_models(self) -> list[str]:
        if self.backend == "vertex":
            return []
        try:
            return [m.id for m in self.client.models.list(limit=100).data]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic: could not list models: {exc}") from exc

    def complete(self, text: str, image_png: bytes | None = None,
                 system: str | None = None, max_tokens: int = 16) -> Response:
        content: list[dict] = []
        if image_png is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_png).decode(),
                },
            })
        content.append({"type": "text", "text": text})

        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        msg = None
        error = None
        with timed() as t:
            try:
                msg = self.client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            return Response("", None, None, t.ms, error=error)

        out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return Response(
            text=out,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=t.ms,
            raw={"id": getattr(msg, "id", None), "stop_reason": msg.stop_reason},
        )
