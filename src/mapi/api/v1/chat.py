"""Chat grounded in a space's memories."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope
from ...domain.synthesis.chat import Turn
from ..deps import Principal, ServiceDep, require_scope
from ..schemas import ChatCitation, ChatRequest, ChatResponse

router = APIRouter(prefix="/spaces/{space_id}", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Chat over this space")
async def chat(
    space_id: str,
    body: ChatRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SEARCH))],
) -> ChatResponse:
    """Answer a question from this space's memories, with citations.

    The answer is grounded: the model sees the retrieved memories and nothing
    else, is told to cite the ones it uses, and is told to decline rather than
    fill a gap. `citations` reports which memories the answer actually leaned
    on, and `considered` reports everything it was shown — so a caller can see
    both what was used and what was passed over.

    A memory carrying `confidence: unverified` in its metadata is labelled as
    such in the model's context, and `used_unverified` flags an answer that
    rests on one.

    Requires a configured synthesis backend. Without one this returns **502
    provider_error** -- the docstring said 503, which is what `StoreError` uses;
    this text is the endpoint's OpenAPI description, so the wrong number was the
    documented contract. Note also that the response `detail` is the generic
    "Upstream provider failed": the error handler replaces `detail` with `title`
    for every status >= 500 unless `debug_errors` is on, so the actionable message
    ("set synthesis_backend=gemini") appears in the server log and not in the
    response.
    """
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")

    answer, response = await service.chat(
        principal.org_id,
        space_id,
        message=body.message,
        history=[Turn(role=t.role, content=t.content) for t in body.history],
        k=body.k,
        remember=body.remember,
    )

    # Cited first and in citation order, then the rest: the caller usually
    # wants to render "sources" and would otherwise have to re-sort.
    by_id = {hit.memory.id: hit for hit in response.results}
    ordered = [by_id[i] for i in answer.cited if i in by_id]
    ordered += [h for h in response.results if h.memory.id not in set(answer.cited)]

    return ChatResponse(
        reply=answer.reply,
        used_unverified=answer.used_unverified,
        intent=str(response.intent.kind) if response.intent else None,
        citations=[
            ChatCitation(
                id=hit.memory.id,
                content=hit.memory.content,
                score=round(hit.score, 4),
                occurred_at=hit.memory.occurred_at,
                tags=list(hit.memory.tags),
                cited=hit.memory.id in set(answer.cited),
                confidence=str((hit.memory.metadata or {}).get("confidence") or ""),
            )
            for hit in ordered
        ],
        conflicts=[list(pair) for pair in response.conflicts],
    )
