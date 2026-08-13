"""`POST /v1/spaces/{id}/chat` — the route no test had ever called.

The one endpoint where the memory system speaks in its own voice, and the whole
suite reached it zero times. Its docstring promises four things a caller depends
on and none of which fail loudly:

  * the answer is grounded -- the model sees the retrieved memories and nothing
    else;
  * `citations` reports what it leaned on, `considered` what it was shown, so a
    caller can see both what was used and what was passed over;
  * cited memories come FIRST and in citation order, so rendering "sources" needs
    no re-sorting;
  * `used_unverified` flags an answer resting on a memory marked unverified.

Without a synthesis backend it must fail cleanly rather than answer from nothing,
which is the first test here because it is the state the app ships in by default.

The `app_context` fixture wires a real app over `InMemoryStore` and a
deterministic embedder, so the completer is injected per test -- no vendor call,
and the reply is whatever the test says it is.
"""

from __future__ import annotations

import httpx
import pytest

FACTS = [
    "Our primary database is Postgres 16 running on Cloud SQL in us-central1.",
    "Redis 7 backs the rate limiter across every process.",
    "The office lease renews in June and the rent rises 4 percent.",
]


async def _seed(client: httpx.AsyncClient, space_id: str, *facts: str) -> list[str]:
    ids = []
    for content in facts or FACTS:
        response = await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": content}
        )
        assert response.status_code == 201, response.text
        ids.append(response.json()["memory"]["id"])
    return ids


def _install(app_context: dict, reply: str) -> None:
    """Inject a completer into the running app's service."""

    async def complete(prompt: str) -> str:
        return reply

    app_context["app"].state.service.completer = complete


# -- the shipped default ---------------------------------------------------


async def test_chat_without_a_backend_fails_cleanly(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """The state the app ships in.

    `synthesis_backend` defaults off, so this is what a caller hits first. It has
    to be a clear provider error -- answering from no model would mean inventing
    the answer, and a 500 would read as our bug rather than as missing config.
    """
    app_context["app"].state.service.completer = None
    response = await client.post(f"/v1/spaces/{space_id}/chat", json={"message": "hello"})
    assert response.status_code == 502, response.text
    body = response.json()
    assert body["code"] == "provider_error"
    # The detail is the GENERIC title, not the actionable message. The error
    # handler replaces `detail` with `title` for every status >= 500 unless
    # `debug_errors` is set, so "set synthesis_backend=gemini" reaches the server
    # log and not the caller. Deliberate for 5xx -- pinned here because it means
    # an operator misconfiguring this endpoint gets "Upstream provider failed"
    # and has to go read logs to find out why.
    assert body["detail"] == "Upstream provider failed"


# -- grounding and citations ----------------------------------------------


async def test_a_grounded_answer_reports_what_it_cited(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    ids = await _seed(client, space_id)
    _install(app_context, "Postgres 16 on Cloud SQL [1]")

    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "what database do we use?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "Postgres 16 on Cloud SQL [1]"
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["id"] in ids


async def test_cited_memories_come_first_and_in_citation_order(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """So a caller rendering "sources" does not have to re-sort.

    The route puts cited hits first in the order the model named them, then
    everything else it was shown. A caller that trusted document order would
    otherwise attribute the answer to whatever ranked highest.
    """
    await _seed(client, space_id)
    _install(app_context, "per [3] and then [1]")

    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "tell me about the setup", "k": 3}
    )
    body = response.json()
    cited = [c["id"] for c in body["citations"] if c["cited"]]
    assert len(cited) == 2
    # The first two entries must be the cited ones, in the model's order.
    assert [c["id"] for c in body["citations"][:2]] == cited


async def test_considered_shows_what_the_model_was_passed_over(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """Uncited memories still appear, flagged.

    "The model saw five and used one" is a different claim from "the model saw
    one", and only the first lets a caller judge a thin answer.
    """
    await _seed(client, space_id)
    _install(app_context, "only the first matters [1]")

    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "what do we run?", "k": 3}
    )
    body = response.json()
    flags = [c["cited"] for c in body["citations"]]
    assert flags.count(True) == 1
    assert False in flags, "memories shown but not used must still be reported"


async def test_an_empty_space_declines_rather_than_inventing(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """No memories, so the model is never called and the reply is the honest one.

    A chat that invents an answer from an empty result set is worse than one that
    returns nothing, because the failure is invisible.
    """
    _install(app_context, "I know all about your database")
    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "what database do we use?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "don't have" in body["reply"]


# -- unverified memories --------------------------------------------------


async def test_an_answer_resting_on_an_unverified_memory_is_flagged(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """The promise that matters most.

    A memory system that launders "someone told me this once" into a confident
    answer is actively harmful. The caveat has to reach the caller, not just the
    model's context.
    """
    created = await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={
            "content": "Revenue was around 4.2 million last quarter.",
            "metadata": {"confidence": "unverified"},
        },
    )
    assert created.status_code == 201, created.text
    _install(app_context, "About 4.2M, though unverified [1]")

    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "what was revenue?"}
    )
    body = response.json()
    assert body["used_unverified"] is True


async def test_an_answer_over_verified_memories_is_not_flagged(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    await _seed(client, space_id)
    _install(app_context, "Postgres 16 [1]")
    response = await client.post(
        f"/v1/spaces/{space_id}/chat", json={"message": "what database?"}
    )
    assert response.json()["used_unverified"] is False


# -- remember, and the request contract -----------------------------------


async def test_remember_defaults_off(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    """A question stored as a memory is a claim nobody made."""
    await _seed(client, space_id)
    _install(app_context, "yes [1]")

    await client.post(
        f"/v1/spaces/{space_id}/chat",
        json={"message": "is Krishna on an F-1 visa?"},
    )
    listing = await client.get(f"/v1/spaces/{space_id}/memories?limit=50")
    contents = [m["content"] for m in listing.json()["items"]]
    assert "is Krishna on an F-1 visa?" not in contents


async def test_remember_true_stores_the_message(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    await _seed(client, space_id)
    _install(app_context, "noted [1]")

    await client.post(
        f"/v1/spaces/{space_id}/chat",
        json={"message": "I moved the standup to 10:15am.", "remember": True},
    )
    listing = await client.get(f"/v1/spaces/{space_id}/memories?limit=50")
    contents = [m["content"] for m in listing.json()["items"]]
    assert "I moved the standup to 10:15am." in contents


async def test_history_is_accepted(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    await _seed(client, space_id)
    _install(app_context, "Redis [2]")
    response = await client.post(
        f"/v1/spaces/{space_id}/chat",
        json={
            "message": "and the cache?",
            "history": [{"role": "user", "content": "what database?"}],
        },
    )
    assert response.status_code == 200, response.text


async def test_a_malformed_space_id_is_a_422(
    client: httpx.AsyncClient, app_context: dict
) -> None:
    _install(app_context, "x")
    response = await client.post("/v1/spaces/not-an-id/chat", json={"message": "hi"})
    assert response.status_code == 422
    assert response.json()["field"] == "space_id"


async def test_an_empty_message_is_rejected(
    client: httpx.AsyncClient, space_id: str, app_context: dict
) -> None:
    _install(app_context, "x")
    response = await client.post(f"/v1/spaces/{space_id}/chat", json={"message": "   "})
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/v1/spaces/{}/chat"])
async def test_chat_requires_authentication(
    anon_client: httpx.AsyncClient, space_id: str, path: str
) -> None:
    """It reads memories, so it needs the same key every read path needs."""
    response = await anon_client.post(path.format(space_id), json={"message": "hi"})
    assert response.status_code == 401
