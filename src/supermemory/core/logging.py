"""Structured logging with request correlation.

Every log line carries the request id, so a single failing call can be pulled
out of a busy log by one grep. Secrets are scrubbed by a processor rather than
by asking every call site to remember, because the call site that forgets is
the one that logs the API key.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
org_id_var: ContextVar[str | None] = ContextVar("org_id", default=None)

_SENSITIVE_KEYS = frozenset(
    {
        "authorization", "api_key", "apikey", "token", "secret", "password",
        "pepper", "x-api-key", "key_plaintext", "gemini_api_key", "openai_api_key",
    }
)
_REDACTED = "[redacted]"


def _scrub(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
    for key in list(event):
        if key.lower() in _SENSITIVE_KEYS:
            event[key] = _REDACTED
    return event


def _add_context(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
    rid = request_id_var.get()
    if rid and "request_id" not in event:
        event["request_id"] = rid
    oid = org_id_var.get()
    if oid and "org_id" not in event:
        event["org_id"] = oid
    return event


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_context,
            _scrub,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s", stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    for noisy in ("uvicorn.access", "uvicorn.error", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "supermemory") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


__all__ = ["configure_logging", "get_logger", "org_id_var", "request_id_var"]
