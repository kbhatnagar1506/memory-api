"""Query expansion.

Dense retrieval matches a *question* against *statements*, and those are written
in different vocabularies. LoCoMo makes the failure concrete: "What fields would
Caroline be likely to pursue in her education?" must find a turn where Caroline
mentions psychology and a counseling certification. The question and the answer
share almost no content words, so cosine similarity between them is weak, and
`open_domain` questions retrieved evidence only 38% of the time against 82% for
questions whose wording overlaps the dialogue.

HyDE (Hypothetical Document Embeddings) closes that gap by embedding a
*hypothetical answer* instead of, or alongside, the question. The hypothesis
does not need to be factually correct — it only needs to be written in the
register of the corpus, so that it lands near the real answer in vector space.

The generated text is averaged with the real query vector rather than replacing
it. A hypothesis that goes off-topic would otherwise drag retrieval with it;
averaging bounds the damage to half the signal, and the query vector keeps the
search anchored to what was actually asked.

Every failure path degrades to the plain query. Expansion is an enhancement, and
an enhancement that can fail a search is a liability.
"""

from __future__ import annotations

import abc
import asyncio

from ...core.logging import get_logger

log = get_logger(__name__)

_PROMPT = """\
Write a short passage that would plausibly appear in a personal conversation \
and that would answer this question. Write it as a statement, in the first \
person, as a person would say it in chat. Do not answer with facts you do not \
have -- invent plausible specifics. Two sentences maximum. Output only the \
passage.

Question: {query}

Passage:"""


class QueryExpander(abc.ABC):
    """Produces extra texts to embed alongside the query."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def expand(self, query: str) -> list[str]:
        """Additional texts. An empty list means "no expansion", not an error."""


class NoopExpander(QueryExpander):
    @property
    def name(self) -> str:
        return "none"

    async def expand(self, query: str) -> list[str]:
        return []


class HydeExpander(QueryExpander):
    """Generates one hypothetical answer passage per query."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        timeout_s: float = 12.0,
        project: str | None = None,
        location: str = "global",
        api_key: str | None = None,
        client: object | None = None,
        max_chars: int = 600,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self._client = client
        if client is None:
            self._client = self._build_client(project, location, api_key)

    @staticmethod
    def _build_client(project: str | None, location: str, api_key: str | None) -> object:
        import os

        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY")
        if key:
            return genai.Client(api_key=key)
        proj = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not proj:
            from ...core.errors import ConfigurationError

            raise ConfigurationError(
                "HyDE expansion needs GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for "
                "Vertex AI with ADC."
            )
        return genai.Client(vertexai=True, project=proj, location=location)

    @property
    def name(self) -> str:
        return "hyde"

    async def _call(self, query: str) -> str:
        """Native async — see GeminiEmbedder._call for why not to_thread."""
        from google.genai import types

        response = await self._client.aio.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=_PROMPT.format(query=query),
            config=types.GenerateContentConfig(
                temperature=0.0,
                # Reasoning tokens share the answer budget on thinking models.
                max_output_tokens=768,
                thinking_config=types.ThinkingConfig(thinking_budget=128),
            ),
        )
        return response.text or ""

    async def expand(self, query: str) -> list[str]:
        if not query.strip():
            return []
        try:
            raw = await asyncio.wait_for(self._call(query), timeout=self.timeout_s)
        except TimeoutError:
            log.warning("hyde_timeout", model=self.model)
            return []
        except Exception as exc:
            log.warning("hyde_failed", model=self.model, error=str(exc)[:200])
            return []

        passage = " ".join(raw.split())[: self.max_chars].strip()
        # A model that returns nothing usable must not produce an empty string:
        # the embedder rejects blank input, which would fail the whole search.
        return [passage] if passage else []


__all__ = ["HydeExpander", "NoopExpander", "QueryExpander"]
