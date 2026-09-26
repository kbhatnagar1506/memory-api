"""The completion seam: an API-key route, thinking per model family, and a
write path that calls a model only when asked to.

Three changes, each closing a way the configuration could not express what
a deployment needed:

  * Completion was Vertex-only, so a deployment holding only a Gemini API
    key could embed but never derive, chat or classify a query.
  * Every call sent `thinking_budget`, which the 3.x models answer with 400
    at 0 -- the value extraction and query understanding use.
  * Turning synthesis on for /derive also put a model call on every write.

No network: `google.genai.Client` is replaced by a recorder, and the real
SDK types build the request so a field the SDK would reject still fails.
"""

from __future__ import annotations

from typing import Any

import pytest

from mapi.config import Settings
from mapi.core.errors import ConfigurationError
from mapi.domain.synthesis import completer
from mapi.domain.synthesis.completer import (
    build_completer,
    build_extractor,
    build_understander,
    thinking_level_for,
    uses_thinking_levels,
)

#: The SDK is the `gemini` extra, and CI's test job installs only `.[dev]`.
genai = pytest.importorskip("google.genai")


class _Recorder:
    """Stands in for `genai.Client`: remembers how it was built and called."""

    built: list[dict[str, Any]]
    calls: list[dict[str, Any]]

    def __init__(self, **kwargs: Any) -> None:
        _Recorder.built.append(kwargs)
        self.aio = self
        self.models = self

    async def generate_content(self, **kwargs: Any) -> Any:
        _Recorder.calls.append(kwargs)

        class _Response:
            text = "ok"
            candidates = ()

        return _Response()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    _Recorder.built = []
    _Recorder.calls = []
    monkeypatch.setattr(genai, "Client", _Recorder)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return _Recorder


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "synthesis_backend": "gemini",
        "google_cloud_project": None,
        "gemini_api_key": None,
    }
    return Settings(**{**base, **overrides})


# -- which family takes which knob -------------------------------------------------


@pytest.mark.parametrize(
    ("model", "levels"),
    [
        ("gemini-2.5-flash", False),
        ("gemini-2.5-flash-lite", False),
        ("gemini-2.0-flash", False),
        ("gemini-1.5-pro", False),
        ("models/gemini-2.5-flash", False),
        ("gemini-3.5-flash", True),
        ("gemini-3.5-flash-lite", True),
        ("gemini-3.1-pro-preview", True),
        ("gemini-flash-lite-latest", True),
        ("publishers/google/models/gemini-3.5-flash", True),
    ],
)
def test_the_family_decides_budget_or_level(model: str, levels: bool) -> None:
    assert uses_thinking_levels(model) is levels


@pytest.mark.parametrize(
    ("model", "budget", "level"),
    [
        ("gemini-3.5-flash-lite", 0, "MINIMAL"),
        ("gemini-3.5-flash", 128, "LOW"),
        ("gemini-3.5-flash", 1024, "LOW"),
        ("gemini-3.5-flash", 4096, "MEDIUM"),
        ("gemini-3.5-flash", 20_000, "HIGH"),
        # Pro has no MINIMAL: measured 400 "Thinking level MINIMAL is not supported".
        ("gemini-3.1-pro-preview", 0, "LOW"),
        ("gemini-pro-latest", 0, "LOW"),
    ],
)
def test_budgets_map_onto_levels(model: str, budget: int, level: str) -> None:
    assert thinking_level_for(model, budget) == level


async def test_a_3x_model_is_sent_a_level_and_no_budget(recorder) -> None:
    """The measured failure: gemini-3.5-flash-lite answers budget 0 with 400."""
    extractor = build_extractor(
        _settings(
            gemini_api_key="k",
            write_extraction=True,
            extraction_model="gemini-3.5-flash-lite",
            extraction_thinking_budget=0,
        )
    )
    assert extractor is not None
    assert await extractor("prompt") == "ok"
    thinking = recorder.calls[0]["config"].thinking_config
    assert str(thinking.thinking_level).endswith("MINIMAL")
    assert thinking.thinking_budget is None


async def test_a_2x_model_is_still_sent_its_budget(recorder) -> None:
    complete = build_completer(
        _settings(google_cloud_project="p", synthesis_model="gemini-2.5-flash")
    )
    assert complete is not None
    await complete("prompt")
    thinking = recorder.calls[0]["config"].thinking_config
    assert thinking.thinking_budget == 128
    assert thinking.thinking_level is None


# -- which credential ----------------------------------------------------------------


def test_an_api_key_alone_is_enough(recorder) -> None:
    assert build_completer(_settings(gemini_api_key="from-settings")) is not None
    assert recorder.built[-1]["api_key"] == "from-settings"
    assert "vertexai" not in recorder.built[-1]


def test_the_sdks_own_key_variable_is_honoured(recorder, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert build_completer(_settings()) is not None
    assert recorder.built[-1]["api_key"] == "from-env"


def test_a_project_keeps_vertex_even_beside_a_key(recorder) -> None:
    """No configuration that worked before changes route: the Developer API
    does not serve every model Vertex does (gemini-2.5-flash: 404)."""
    build_completer(_settings(google_cloud_project="p", gemini_api_key="k"))
    built = recorder.built[-1]
    assert built["vertexai"] is True
    assert built["project"] == "p"
    assert "api_key" not in built


def test_no_credential_at_all_is_a_configuration_error(recorder) -> None:
    with pytest.raises(ConfigurationError, match="gemini_api_key"):
        build_completer(_settings())


# -- the write path calls a model only when asked ------------------------------------


def test_synthesis_alone_no_longer_puts_a_model_on_every_write(recorder) -> None:
    settings = _settings(google_cloud_project="p")
    assert settings.write_extraction is False
    assert build_completer(settings) is not None, "derive and chat still get their model"
    assert build_extractor(settings) is None, "writes do not"


def test_write_extraction_turns_the_write_path_model_on(recorder) -> None:
    assert build_extractor(_settings(google_cloud_project="p", write_extraction=True))


def test_write_extraction_without_a_backend_is_inert(recorder) -> None:
    assert build_extractor(Settings(write_extraction=True)) is None


def test_understanding_is_unaffected_by_the_write_switch(recorder) -> None:
    assert build_understander(_settings(google_cloud_project="p")) is not None


def test_the_module_exports_its_helpers() -> None:
    assert {"thinking_level_for", "uses_thinking_levels"} <= set(completer.__all__)
