"""OpenAI adapter.

OpenAI has no Application Default Credentials path, so this provider requires
OPENAI_API_KEY. If the key is absent the adapter refuses to construct rather
than silently dropping the arm from the matrix.
"""

from __future__ import annotations

import base64
import os

from .base import ProviderError, Response, resolve_model, timed

NAME = "openai"


class OpenAIAdapter:
    name = NAME

    def __init__(self, model_candidates: list[str], temperature: float | None = 0.0):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderError(
                "OPENAI_API_KEY is not set. OpenAI has no ADC path, so this "
                "provider needs a key or must be dropped from --models."
            )
        self.client = OpenAI()
        self.temperature = temperature
        self.model = resolve_model(model_candidates, self.list_models(), NAME)
        self.model_resolution = "resolved against live models.list()"

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"openai: could not list models: {exc}") from exc

    def complete(self, text: str, image_png: bytes | None = None,
                 system: str | None = None, max_tokens: int = 16) -> Response:
        content: list[dict] = []
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            })
        content.append({"type": "input_text", "text": text})

        kwargs: dict = dict(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max(16, max_tokens),
        )
        if system:
            kwargs["instructions"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        resp = None
        error = None
        with timed() as t:
            try:
                resp = self.client.responses.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            return Response("", None, None, t.ms, error=error)

        usage = getattr(resp, "usage", None)
        return Response(
            text=(getattr(resp, "output_text", "") or ""),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=t.ms,
            raw={"id": getattr(resp, "id", None), "status": getattr(resp, "status", None)},
        )
