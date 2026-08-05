"""Provider registry: name -> adapter factory.

Model IDs are never hardcoded from memory. Config supplies *candidates* (a
prefix or exact ID), and each adapter resolves them against the provider's live
model list where one exists, recording both the resolved ID and how it was
resolved in the run manifest.
"""

from __future__ import annotations

from .base import ProviderAdapter, ProviderError, Response  # noqa: F401

PROVIDERS = {
    "anthropic": "mapi_ablation.providers.anthropic_p:AnthropicAdapter",
    "gemini": "mapi_ablation.providers.gemini_p:GeminiAdapter",
    "openai": "mapi_ablation.providers.openai_p:OpenAIAdapter",
}


def make_adapter(provider: str, model_candidates: list[str],
                 temperature: float | None = 0.0, **kw) -> ProviderAdapter:
    import importlib

    if provider not in PROVIDERS:
        raise ProviderError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
    mod_name, cls_name = PROVIDERS[provider].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(model_candidates, temperature=temperature, **kw)
