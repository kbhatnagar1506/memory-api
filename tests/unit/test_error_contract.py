"""The error hierarchy as a contract, including the parts nothing raises.

Two exception classes -- `IdempotencyConflictError` and
`UnsupportedMediaTypeError` -- are defined, exported, and raised nowhere in
`src/`. That is not automatically wrong: a public error taxonomy can legitimately
name a status the API reserves. But an unraised error is indistinguishable from a
forgotten one, and either it should be wired up or deleted. This file asserts the
current state so the decision is explicit rather than pending forever.

The rest pins what a caller actually parses: the problem-document shape, the
status of every class, and the two places the handler deliberately rewrites the
response -- `retry-after` on 429, and `detail` replaced by `title` on any 5xx.
That second one is worth knowing about, because it is why a misconfigured chat
backend returns "Upstream provider failed" instead of the fix.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from mapi.core import errors as errors_module
from mapi.core.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    IdempotencyConflictError,
    MapiError,
    NotFoundError,
    PayloadTooLargeError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    StoreError,
    UnauthorizedError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from mapi.core.quota import QuotaExceededError

#: Every error class with the status a caller will see. A change here is a
#: change to the public contract, so it should be a visible diff.
STATUSES: tuple[tuple[type[MapiError], int, str], ...] = (
    (MapiError, 500, "internal_error"),
    (ValidationError, 422, "validation_error"),
    (BadRequestError, 400, "bad_request"),
    (UnauthorizedError, 401, "unauthorized"),
    (ForbiddenError, 403, "forbidden"),
    (NotFoundError, 404, "not_found"),
    (ConflictError, 409, "conflict"),
    (IdempotencyConflictError, 409, "idempotency_conflict"),
    (PayloadTooLargeError, 413, "payload_too_large"),
    (UnsupportedMediaTypeError, 415, "unsupported_media_type"),
    (RateLimitedError, 429, "rate_limited"),
    (ProviderError, 502, "provider_error"),
    (ProviderTimeoutError, 504, "provider_timeout"),
    (StoreError, 503, "store_unavailable"),
    (QuotaExceededError, 402, "quota_exceeded"),
)


@pytest.mark.parametrize(
    ("cls", "status", "slug"), STATUSES, ids=[c.__name__ for c, _s, _g in STATUSES]
)
def test_each_error_carries_its_status_and_slug(
    cls: type[MapiError], status: int, slug: str
) -> None:
    error = (
        cls("at the memory limit", limit_name="memories", limit=10, current=10)
        if cls is QuotaExceededError
        else cls("something went wrong")
    )
    assert error.status_code == status
    assert error.slug == slug


def test_the_problem_document_has_the_fields_a_client_parses() -> None:
    """RFC 7807. `type` is a stable URL, `code` is the machine-readable slug."""
    problem = NotFoundError("no such memory", field="memory_id").to_problem()
    assert problem["status"] == 404
    assert problem["code"] == "not_found"
    assert problem["type"].endswith("/not_found")
    assert problem["field"] == "memory_id"
    assert problem["detail"] == "no such memory"
    assert "title" in problem


def test_extra_payload_is_merged_into_the_document() -> None:
    """How `QuotaExceededError` reports which limit was hit -- a caller needs the
    number to know whether to retry or to upgrade."""
    problem = QuotaExceededError(
        "too much stored", limit_name="bytes_stored", limit=1000, current=1200
    ).to_problem()
    assert problem["limit_name"] == "bytes_stored"
    assert problem["limit"] == 1000
    assert problem["current"] == 1200


def test_rate_limited_reports_its_retry_after() -> None:
    problem = RateLimitedError(retry_after=2.5).to_problem()
    assert problem["retry_after"] == 2.5


def test_quota_exceeded_is_the_only_error_outside_the_errors_module() -> None:
    """It lives in `core/quota.py` beside the thing that raises it.

    Worth pinning: a reader looking for the 402 will not find it in
    `core/errors.py`, and a second stray subclass elsewhere would make the
    taxonomy impossible to enumerate.
    """
    assert QuotaExceededError.__module__ == "mapi.core.quota"
    declared = {
        name
        for name, obj in inspect.getmembers(errors_module, inspect.isclass)
        if issubclass(obj, MapiError)
    }
    from_repo: set[str] = set()
    root = Path(__file__).resolve().parents[2] / "src"
    for path in root.rglob("*.py"):
        for match in re.finditer(
            r"^class (\w+)\((?:\w+\.)?(\w*Error)\)", path.read_text(), re.M
        ):
            name, base = match.groups()
            if base in declared or base == "MapiError":
                from_repo.add(name)
    stray = from_repo - declared - {"QuotaExceededError"}
    assert not stray, f"MapiError subclasses outside core/errors.py: {sorted(stray)}"


# -- the two nobody raises -------------------------------------------------


@pytest.mark.parametrize(
    "cls", [IdempotencyConflictError, UnsupportedMediaTypeError], ids=lambda c: c.__name__
)
def test_an_unraised_error_is_recorded_as_unraised(cls: type[MapiError]) -> None:
    """Defined, exported, raised nowhere. Deliberate or forgotten -- decide.

    An error class with no raise site reserves a status and a slug in the public
    taxonomy while being unreachable, so a client cannot ever receive it and a
    reader cannot tell whether that is intentional. This test fails the moment
    someone raises one, which is the right moment to delete the test; and it keeps
    failing to be true if someone deletes the class, which is the other
    resolution.
    """
    root = Path(__file__).resolve().parents[2] / "src"
    raisers = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if re.search(rf"\braise {cls.__name__}\b", path.read_text())
    ]
    assert not raisers, (
        f"{cls.__name__} is now raised in {raisers}. Delete this parametrization "
        "and add a test for the behaviour that raises it."
    )


def test_the_unraised_errors_are_still_exported() -> None:
    """If they are staying, they stay reachable -- an unraised AND unexported
    class is just dead code with a docstring."""
    assert "IdempotencyConflictError" in errors_module.__all__
    assert "UnsupportedMediaTypeError" in errors_module.__all__


# -- inheritance, which decides what an `except` clause catches -------------


def test_the_specific_conflicts_are_catchable_as_conflict() -> None:
    """A handler writing `except ConflictError` must catch the idempotency one,
    or adding a subclass silently changes which handler runs."""
    assert issubclass(IdempotencyConflictError, ConflictError)
    assert issubclass(ProviderTimeoutError, ProviderError)


def test_every_error_is_catchable_as_mapi_error() -> None:
    """The single `except MapiError` in the handler is the whole reason the
    taxonomy works. A class outside it becomes an opaque 500."""
    for cls, _status, _slug in STATUSES:
        assert issubclass(cls, MapiError)


def test_a_mapi_error_is_an_exception_not_a_value_error() -> None:
    """Pinned because it is the trap the `list_relations` defect fell into.

    A bare `ValueError` raised from the store is NOT a `MapiError`, so it escapes
    the handler and becomes "an unexpected error occurred". Anything in this
    hierarchy is caught and rendered; anything outside it is a 500.
    """
    assert issubclass(MapiError, Exception)
    assert not issubclass(MapiError, ValueError)
    assert not issubclass(ValueError, MapiError)
