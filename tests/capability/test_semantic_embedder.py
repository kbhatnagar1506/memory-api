"""The one thing constructed vectors cannot test: whether an embedder means it.

Every other capability family specifies its own geometry, because the question is
what the PIPELINE does with a given similarity structure. Exactly one question
needs a real model: does a production embedder put a paraphrase closer than a
lexical near-neighbour? That is a property of the provider's weights and cannot be
constructed, mocked, or inferred.

So this is the only file behind `live_llm`, and it is scoped narrowly on purpose.
Putting PIPELINE behaviour behind a credential gate would mean a claim that never
runs -- the suite would look like it covered retrieval while skipping the half that
needs a key. Provider semantics are different: there is no way to check them
without a provider, and no reason for the rest of the suite to wait.

Also covers the dimension contract, because `gemini.py` and `openai.py` are at 0%
and the failure mode they share is the expensive one: a model or config change that
returns a different width silently corrupts an existing index. `_validate` raises
on that, and this is where it gets exercised against something real.

    pytest tests/capability -m live_llm        # needs credentials
    pytest tests/capability -m "not live_llm"  # the default lane
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live_llm

#: The model under test, from the environment the service itself reads. The
#: question this file asks belongs to one model, and the hard-coded default
#: (text-embedding-004) was retired underneath it -- it now answers 404 on the
#: Developer API, so the lane failed on the provider rather than on semantics.
MODEL = os.getenv("MAPI_EMBEDDING_MODEL") or "gemini-embedding-001"
#: The service's own key variable first, then the SDK's conventional one.
_API_KEY = os.getenv("MAPI_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or None

#: Skipped rather than failed: absent credentials are the normal case for anyone
#: who is not deliberately running this lane.
_HAS_CREDENTIALS = bool(_API_KEY or os.getenv("GOOGLE_CLOUD_PROJECT"))
pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not _HAS_CREDENTIALS,
        reason="needs MAPI_GEMINI_API_KEY / GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex",
    ),
]

DIMENSIONS = 768


def _embedder() -> object:
    from mapi.domain.embeddings.gemini import GeminiEmbedder

    return GeminiEmbedder(model=MODEL, dimensions=DIMENSIONS, api_key=_API_KEY)


async def test_a_real_embedder_prefers_meaning_over_spelling() -> None:
    """The inverse of `test_the_repo_embedder_is_lexical_not_semantic`.

    `DeterministicEmbedder` puts "car" closer to "cart" than to "automobile",
    which is why the capability families construct their vectors. A production
    embedder must do the opposite -- and if it does not, tier T2 of the
    lexical-overlap ladder (zero shared content words, high cosine) does not
    describe anything achievable in production and the pipeline's vector arm is
    not doing the job it is configured for.
    """
    from mapi.domain.embeddings.base import cosine_similarity

    embedder = _embedder()
    car, automobile, cart = (
        await embedder.embed(["a car", "an automobile", "a shopping cart"])  # type: ignore[attr-defined]
    ).vectors

    assert cosine_similarity(car, automobile) > cosine_similarity(car, cart), (
        "the provider ranks a shared prefix above a synonym -- retrieval's vector "
        "arm is behaving lexically and the whole hybrid design is moot"
    )


async def test_a_paraphrase_with_no_shared_content_words_is_still_close() -> None:
    """Tier T2, against a real model.

    The families assert that the PIPELINE surfaces a memory at cosine 0.80 with
    zero lexical overlap. This asserts such memories can exist -- that a real
    embedder actually produces a high cosine for a genuine paraphrase sharing no
    content words, which is the premise the whole vector arm rests on.
    """
    from mapi.domain.embeddings.base import cosine_similarity

    embedder = _embedder()
    question, paraphrase, unrelated = (
        await embedder.embed(  # type: ignore[attr-defined]
            [
                "Which seat do I prefer on long flights?",
                "On lengthy journeys I always pick a spot beside the window.",
                "The quarterly invoice from the monitoring vendor was approved.",
            ]
        )
    ).vectors

    assert cosine_similarity(question, paraphrase) > cosine_similarity(question, unrelated)


async def test_the_query_side_transform_is_used_and_differs() -> None:
    """`embed_query` exists because asymmetric embedding measurably helps recall.

    `GeminiEmbedder` sets RETRIEVAL_QUERY for queries and RETRIEVAL_DOCUMENT for
    documents. The pipeline prefers `embed_query` when a provider defines it, so a
    provider that defined it and returned the document-side vector would silently
    lose that gain -- present in the config, absent in the behaviour.
    """
    from mapi.domain.embeddings.base import cosine_similarity

    embedder = _embedder()
    text = "Which seat do I prefer on long flights?"
    as_query = await embedder.embed_query(text)  # type: ignore[attr-defined]
    as_document = (await embedder.embed([text])).vectors[0]  # type: ignore[attr-defined]

    assert len(as_query) == len(as_document) == DIMENSIONS
    assert cosine_similarity(as_query, as_document) < 0.9999, (
        "embed_query returned the document-side vector; the asymmetric task type "
        "is configured but not taking effect"
    )


async def test_the_provider_honours_the_requested_dimension() -> None:
    """The failure mode that corrupts an index rather than one query.

    `_validate` raises a `ProviderError` naming this explicitly -- "a model or
    config change of this kind silently corrupts an existing index" -- because a
    width change makes every stored vector incomparable while every individual
    call still looks successful.
    """
    embedder = _embedder()
    vectors = (await embedder.embed(["one", "two"])).vectors  # type: ignore[attr-defined]
    assert [len(v) for v in vectors] == [DIMENSIONS, DIMENSIONS]


async def test_returned_vectors_are_finite_and_normalized() -> None:
    """The contract every downstream cosine assumes.

    A non-finite value poisons every comparison it touches, and a zero vector
    makes cosine undefined. `_validate` rejects both and re-normalizes the rest,
    so this checks the invariant holds against a real response rather than a stub.
    """
    import math

    embedder = _embedder()
    for vector in (await embedder.embed(["a memory", "another memory"])).vectors:  # type: ignore[attr-defined]
        assert all(math.isfinite(x) for x in vector)
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-6)


async def test_a_blank_input_is_rejected_before_the_network() -> None:
    """Most providers return a zero or arbitrary vector for whitespace, and that
    becomes a memory matching everything equally. Rejected locally, so it costs
    no call and fails at the caller rather than in the index."""
    embedder = _embedder()
    with pytest.raises(ValueError, match="empty or whitespace"):
        await embedder.embed(["   "])  # type: ignore[attr-defined]
