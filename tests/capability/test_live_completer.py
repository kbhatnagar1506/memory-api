"""The completion seam against the real Developer API, with only an API key.

The unit tests (tests/unit/test_completer.py) prove what is SENT; this proves
the provider accepts it. The case that motivated it is the one asserted
hardest: a 3.x flash-lite model at thinking budget 0 -- what extraction and
query understanding ask for -- was a 400 INVALID_ARGUMENT when sent as a
budget, and must succeed now that it is sent as a level.

Skipped without a key, like the rest of the live lane.
"""

from __future__ import annotations

import os

import pytest

#: The service's own key variable first, then the SDK's conventional one.
_API_KEY = os.getenv("MAPI_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or None
#: The 2.5 family is retired on the Developer API (404), so the lane defaults
#: to the current flash-lite and flash; override to probe another model.
LITE = os.getenv("MAPI_LIVE_LITE_MODEL") or "gemini-3.5-flash-lite"
FLASH = os.getenv("MAPI_LIVE_FLASH_MODEL") or "gemini-3.5-flash"

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not _API_KEY, reason="needs MAPI_GEMINI_API_KEY or GEMINI_API_KEY"),
]


def _settings(**overrides: object):
    from mapi.config import Settings

    return Settings(
        synthesis_backend="gemini",
        google_cloud_project=None,
        gemini_api_key=_API_KEY,
        **overrides,
    )


async def test_flash_lite_at_budget_zero_is_accepted_as_a_level() -> None:
    from mapi.domain.synthesis.completer import build_extractor

    extract = build_extractor(
        _settings(write_extraction=True, extraction_model=LITE, extraction_thinking_budget=0)
    )
    assert extract is not None
    answer = await extract("Reply with exactly one word, the word ok, in lower case.")
    assert "ok" in answer.lower()


async def test_the_read_path_completer_answers_on_an_api_key_alone() -> None:
    from mapi.domain.synthesis.completer import build_completer

    complete = build_completer(_settings(synthesis_model=FLASH))
    assert complete is not None
    answer = await complete("What is 2 + 3? Reply with the digit only.")
    assert "5" in answer
