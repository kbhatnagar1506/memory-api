"""Supermemory.ai as a benchmark provider, measured on our terms.

Why this exists: every published number in this category is self-reported, and
the one case with an independent check showed a 45-point gap (Mem0: 94.4%
claimed, 49.0% measured). The fix is not to distrust vendors — it is to run
them through the SAME harness, the SAME official judge prompts, and the SAME
cross-family grader as ourselves, so the only variable is the memory system.

What this measures and what it does NOT:
  * It measures their RETRIEVAL as delivered through their public API, scored
    against LongMemEval's labelled evidence, then answered and graded by our
    pipeline.
  * It does NOT measure their end-to-end product. Supermemory ships its own
    answering layer, profiles, and "infinite chat" proxy; none of that is in
    the loop here. A lower number than they publish is therefore expected and
    is not evidence of a worse product.
  * Their ingest is ASYNCHRONOUS (`status: queued` -> `done`), so anything
    that queries before processing completes measures a race, not a system.
    Ingestion polls to completion before a single query is issued.

API surface, established empirically (their docs point at /v4/search, which
returns 0 results for every query on this account; /v3/search works):

    POST /v3/documents  {content, containerTag}   -> {id, status}
    GET  /v3/documents/{id}                       -> {..., status: done}
    POST /v3/search     {q, containerTags, limit} -> {results: [{documentId,
                                                     chunks: [{content, score}]}]}
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import httpx

BASE_URL = "https://api.supermemory.ai"


@dataclass
class SupermemoryIngest:
    """What was loaded, so a query never runs against a half-built index."""

    #: our document id -> their document id
    doc_ids: dict[str, str] = field(default_factory=dict)
    #: corpus_id -> containerTag
    containers: dict[str, str] = field(default_factory=dict)
    submitted: int = 0
    completed: int = 0
    failed: int = 0


class SupermemoryClient:
    """Thin async client. Native async only -- the blocking-client-under-
    `to_thread` pattern deadlocked a 500-corpus run in this project once."""

    def __init__(self, api_key: str | None = None, *, concurrency: int = 8) -> None:
        key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        if not key:
            raise RuntimeError("SUPERMEMORY_API_KEY is required")
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(120.0),
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def add(self, content: str, container_tag: str) -> str | None:
        async with self._sem:
            for attempt in range(3):
                try:
                    r = await self._client.post(
                        "/v3/documents",
                        json={"content": content, "containerTag": container_tag},
                    )
                    if r.status_code == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if r.status_code >= 300:
                        return None
                    return str(r.json().get("id") or "")
                except Exception:
                    await asyncio.sleep(1 + attempt)
            return None

    async def status(self, doc_id: str) -> str:
        async with self._sem:
            try:
                r = await self._client.get(f"/v3/documents/{doc_id}")
                return (
                    str(r.json().get("status", "unknown")) if r.status_code < 300 else "error"
                )
            except Exception:
                return "error"

    async def search(self, query: str, container_tag: str, limit: int = 10) -> list[dict]:
        """Ranked documents. Their chunk scores are collapsed to a document
        score by taking the max, which is how our own pipeline maps chunk hits
        back to memories -- keeping the comparison structural rather than
        favouring either scoring convention."""
        async with self._sem:
            for attempt in range(3):
                try:
                    r = await self._client.post(
                        "/v3/search",
                        json={"q": query, "containerTags": [container_tag], "limit": limit},
                    )
                    if r.status_code == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if r.status_code >= 300:
                        return []
                    out = []
                    for item in r.json().get("results", []):
                        chunks = item.get("chunks") or []
                        out.append(
                            {
                                "document_id": item.get("documentId"),
                                "score": max(
                                    (c.get("score", 0.0) for c in chunks), default=0.0
                                ),
                                "content": "\n".join(c.get("content", "") for c in chunks),
                            }
                        )
                    return out
                except Exception:
                    await asyncio.sleep(1 + attempt)
            return []


async def wait_until_indexed(
    client: SupermemoryClient,
    doc_ids: list[str],
    *,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 900.0,
) -> tuple[int, int]:
    """Block until every document reports `done`, or the deadline passes.

    Querying an asynchronous index before it finishes measures the queue, not
    the memory system. Returns (completed, still_pending) so a run can report
    honestly that it scored against a partially-built index rather than
    silently under-reporting the vendor.
    """
    pending = set(doc_ids)
    waited = 0.0
    while pending and waited < timeout_seconds:
        await asyncio.sleep(poll_seconds)
        waited += poll_seconds
        states = await asyncio.gather(*(client.status(d) for d in list(pending)))
        for doc_id, state in zip(list(pending), states, strict=False):
            if state in {"done", "error"}:
                pending.discard(doc_id)
        print(
            f"    indexing: {len(doc_ids) - len(pending)}/{len(doc_ids)} done ({waited:.0f}s)",
            flush=True,
        )
    return len(doc_ids) - len(pending), len(pending)


__all__ = [
    "SupermemoryClient",
    "SupermemoryIngest",
    "wait_until_indexed",
]
