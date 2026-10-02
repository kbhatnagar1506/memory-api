"""Code and JSON chunk by their structure: a function whole when it fits, cut only between
its units when it doesn't, every piece embedded knowing whose piece it is."""

from __future__ import annotations

import json
import random

from mapi.config import Settings
from mapi.domain.chunking import (
    CONTEXT_TOKENS,
    MAX_STRUCTURED_TOKENS,
    chunk_content,
    chunk_structured,
    chunk_text,
    detect_kind,
)
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

DOES = "Cancels a pending order after the customer confirms which one."


def _function(reads: int, note_words: int = 60) -> str:
    recipe = {
        "does": DOES,
        "reads": [
            {
                "id": f"r{i}",
                "tool": f"get_thing_{i}",
                "args": {"order": {"dep": f"r{i - 1}.orders[*].id"}},
                "note": "word " * note_words,
            }
            for i in range(reads)
        ],
        "guards": [{"step": "r1", "path": "status", "op": "in", "values": ["pending"]}],
        "write": {"tool": "cancel_order", "args": {"order_id": {"select": "r1.orders[*]"}}},
    }
    return f"{DOES}\n\n{json.dumps(recipe, sort_keys=True)}"


def _inside_a_string(text: str, at: int) -> bool:
    quoted, escaped = False, False
    for c in text[:at]:
        if escaped:
            escaped = False
        elif c == "\\":
            escaped = True
        elif c == '"':
            quoted = not quoted
    return quoted


def test_detects_json_code_and_prose() -> None:
    assert detect_kind(_function(2)) == "json"
    assert detect_kind('{"a": 1}') == "json"
    assert detect_kind("Here:\n[1, 2, 3]") == "json"
    assert detect_kind("Notes\n\n```python\ndef f():\n    return 1\n```") == "code"
    assert detect_kind("import os\n\ndef f(x):\n    return x\n") == "code"
    assert detect_kind("I moved to Madrid in May. The flat is near the park.") == "prose"
    assert detect_kind("{not json") == "prose"


def test_a_function_that_fits_is_one_chunk_however_long() -> None:
    content = _function(8)
    assert 320 < len(content) // 4 <= MAX_STRUCTURED_TOKENS  # past the prose window
    chunks = chunk_content(content)
    assert len(chunks) == 1 and chunks[0].text == content and chunks[0].context == ""


def test_a_big_function_is_cut_only_between_its_members() -> None:
    content = _function(40)
    chunks = chunk_content(content)
    assert len(chunks) > 1
    for chunk in chunks:
        assert content[chunk.start : chunk.end] == chunk.text  # an exact span, stored as is
        assert chunk.token_estimate <= MAX_STRUCTURED_TOKENS
        assert not _inside_a_string(content, chunk.start)
        assert not _inside_a_string(content, chunk.end)
    # Each read is whole in exactly one chunk.
    for i in range(40):
        assert sum(f'"id": "r{i}"' in c.text for c in chunks) == 1
    # Every later chunk says whose piece it is, and where in it.
    assert chunks[0].text.startswith(DOES) and DOES not in chunks[0].context
    assert all(c.context.startswith(DOES) and "in reads" in c.context for c in chunks[1:])


def test_a_member_too_big_for_a_chunk_is_opened_up() -> None:
    giant = {"steps": {f"s{i}": {"text": "word " * 300} for i in range(8)}}
    content = "A recipe with very long steps.\n\n" + json.dumps(giant)
    chunks = chunk_structured(content, "json", max_tokens=600)
    assert len(chunks) >= 4
    assert any("in steps" in c.context for c in chunks)
    for chunk in chunks:
        assert content[chunk.start : chunk.end] == chunk.text
        assert not _inside_a_string(content, chunk.start)


def test_code_is_cut_between_definitions_with_their_decorators() -> None:
    defs = "\n\n".join(
        f"@cached\ndef helper_{i}(x):\n    y = x + {i}\n    return y  " + "# pad " * 80
        for i in range(30)
    )
    content = "Helpers for the cart.\n\nimport os\n\n" + defs
    chunks = chunk_content(content)
    assert len(chunks) > 1
    for chunk in chunks:
        assert content[chunk.start : chunk.end] == chunk.text
        first = chunk.text.splitlines()[0]
        assert first.startswith(("@cached", "Helpers")), first  # never mid-function
    for i in range(30):
        assert sum(f"def helper_{i}(" in c.text for c in chunks) == 1
    assert all(c.context == "Helpers for the cart." for c in chunks[1:])


def test_fenced_code_blocks_stay_whole() -> None:
    block = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(200)) + "\n```"
    content = "\n\n".join(["Setup notes. " * 200, block, "More notes. " * 200])
    chunks = chunk_content(content)
    assert sum(block in c.text for c in chunks) == 1


def test_one_enormous_function_still_splits_and_loses_nothing() -> None:
    body = "\n".join(f"    step_{i} = call_{i}(step_{i - 1})" for i in range(800))
    content = f"def pipeline(x):\n{body}\n    return step_799"
    chunks = chunk_content(content)
    assert len(chunks) > 1
    # Only the whitespace between pieces is not stored.
    assert "".join(c.text for c in chunks).split() == "".join(content).split() or (
        "".join("".join(c.text.split()) for c in chunks) == "".join(content.split())
    )


def test_prose_chunks_exactly_as_before() -> None:
    prose = "I moved to Madrid in May. The flat is near the park. " * 300
    assert chunk_content(prose) == chunk_text(prose)


def test_a_hint_overrides_detection() -> None:
    content = _function(40)
    assert chunk_content(content, kind="prose") == chunk_text(content)


def _random_json(rng: random.Random, depth: int = 0) -> object:
    if depth > 3 or rng.random() < 0.3:
        return rng.choice(
            [
                rng.randint(-9, 99999),
                "".join(rng.choices('ab "\\,{}[]: xyz', k=rng.randint(0, 300))),
                True,
                None,
            ]
        )
    if rng.random() < 0.5:
        return [_random_json(rng, depth + 1) for _ in range(rng.randint(1, 6))]
    return {
        f"k{i}" + ('"x' if rng.random() < 0.2 else ""): _random_json(rng, depth + 1)
        for i in range(rng.randint(1, 6))
    }


def test_any_json_chunks_into_exact_spans_that_cover_it() -> None:
    rng = random.Random(7)
    for _ in range(200):
        value = _random_json(rng)
        if not isinstance(value, dict | list):
            continue
        content = "Describes a value.\n\n" + json.dumps(value)
        chunks = chunk_structured(content, "json", max_tokens=CONTEXT_TOKENS + 64)
        covered: set[int] = set()
        for chunk in chunks:
            assert content[chunk.start : chunk.end] == chunk.text
            covered |= set(range(chunk.start, chunk.end))
        # Only whitespace and the separators between members fall between chunks.
        gaps = {content[i] for i in range(len(content)) if i not in covered}
        assert gaps <= set(" ,{}[]:\n"), gaps


async def test_ingest_keeps_a_function_whole_and_embeds_later_pieces_with_context() -> None:
    settings_ = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=64), HeuristicReranker(), settings_
    )
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    seen: list[str] = []
    original = service.embedder.embed

    async def spy(texts):
        seen.extend(texts)
        return await original(texts)

    service.embedder.embed = spy  # type: ignore[method-assign]
    small = await service.ingest(
        org_id=org.id, space_id=space.id, content=_function(8), extract=False
    )
    assert small.chunk_count == 1
    big = await service.ingest(
        org_id=org.id, space_id=space.id, content=_function(40), extract=False
    )
    assert big.chunk_count > 1
    stored = await service.get_memory(org.id, space.id, big.memory.id)
    assert all(DOES not in c.text or c.ordinal == 0 for c in stored.chunks)  # stored as is
    later = seen[-(big.chunk_count - 1) :]
    assert all(DOES in text and "<context>" in text for text in later)  # embedded with it
