"""Cache-aware concurrent execution.

Three rules that are not negotiable:

  1. A number reaches a results file only if it came from an actual API
     response. Failures are recorded as failures; they are never backfilled,
     retried into a different arm, or replaced with a plausible value.
  2. Every raw response is persisted in full, not just the parsed ID, so any
     claim in the report can be traced back to what the model actually said.
  3. `unparseable` is its own outcome, distinct from `wrong`. An arm that
     answers in prose instead of an ID is failing differently, and collapsing
     the two would hide that.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .providers.base import Response

NODE_ID_RE = re.compile(r"\bN(\d{2})\b")

OUTCOMES = ("correct", "wrong", "unparseable", "error")


@dataclass
class Call:
    """One unit of work: a single (model, arm, condition, question) cell."""

    model_key: str
    provider: str
    model: str
    arm: str
    condition: str
    qid: str
    question: str
    expected: str
    payload_hash: str
    prompt_version: str
    seed: int
    n_nodes: int
    qtype: str
    # payload carried separately so the cache key stays small
    text: str = field(repr=False, default="")
    image_png: bytes | None = field(repr=False, default=None)
    system: str | None = field(repr=False, default=None)
    #: Query covariates (hops, t_gap, pixel_dist, post-resize cap height...).
    #: These ride along so the report can explain WHY an arm lost, not just
    #: that it did. Not part of the cache key.
    meta: dict = field(repr=False, default_factory=dict)

    def cache_key(self) -> str:
        raw = "|".join([
            self.model, self.arm, self.condition, self.qid,
            self.payload_hash, self.prompt_version,
        ])
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class Result:
    call: Call
    response: Response
    parsed: str | None
    outcome: str
    cached: bool = False

    def row(self) -> dict:
        c = self.call
        return {
            "model_key": c.model_key,
            "provider": c.provider,
            "model": c.model,
            "arm": c.arm,
            "condition": c.condition,
            "qid": c.qid,
            "qtype": c.qtype,
            "seed": c.seed,
            "n_nodes": c.n_nodes,
            "question": c.question,
            "expected": c.expected,
            "parsed": self.parsed,
            "outcome": self.outcome,
            "raw_text": self.response.text,
            "input_tokens": self.response.input_tokens,
            "output_tokens": self.response.output_tokens,
            "latency_ms": round(self.response.latency_ms, 1),
            "error": self.response.error,
            "cached": self.cached,
            **c.meta,
        }


def parse_node_id(text: str) -> str | None:
    """First N\\d{2} in the response, or None -> outcome `unparseable`."""
    m = NODE_ID_RE.search(text or "")
    return f"N{m.group(1)}" if m else None


def score_outcome(response: Response, expected: str) -> tuple[str | None, str]:
    if not response.ok:
        return None, "error"
    parsed = parse_node_id(response.text)
    if parsed is None:
        return None, "unparseable"
    return parsed, ("correct" if parsed == expected else "wrong")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """SQLite cache so reruns cost nothing. Keyed exactly as the spec requires."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, text TEXT, input_tokens INT,"
                " output_tokens INT, latency_ms REAL, error TEXT, created TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Response | None:
        cur = self._conn().execute(
            "SELECT text, input_tokens, output_tokens, latency_ms, error "
            "FROM responses WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Response(row[0], row[1], row[2], row[3] or 0.0, error=row[4])

    def put(self, key: str, r: Response) -> None:
        # Errors are not cached: a 429 should not become a permanent result.
        if not r.ok:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)",
                (key, r.text, r.input_tokens, r.output_tokens, r.latency_ms,
                 r.error, datetime.now(timezone.utc).isoformat()),
            )


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


class RetryableError(RuntimeError):
    pass


_RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504", "overloaded", "rate limit",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "Timeout",
)


def _is_retryable(err: str) -> bool:
    return any(m.lower() in err.lower() for m in _RETRYABLE_MARKERS)


@retry(
    retry=retry_if_exception_type(RetryableError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=False,
)
def _call_with_retry(adapter, call: Call, max_tokens: int) -> Response:
    r = adapter.complete(
        text=call.text,
        image_png=call.image_png,
        system=call.system,
        max_tokens=max_tokens,
    )
    if not r.ok and _is_retryable(r.error or ""):
        raise RetryableError(r.error)
    return r


class Runner:
    def __init__(self, adapters: dict[str, object], cache: ResponseCache,
                 concurrency: int = 6, max_tokens: int = 16,
                 responses_path: Path | None = None):
        self.adapters = adapters
        self.cache = cache
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.responses_path = responses_path
        self._write_lock = threading.Lock()

    def _run_one(self, call: Call) -> Result:
        key = call.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            parsed, outcome = score_outcome(cached, call.expected)
            return Result(call, cached, parsed, outcome, cached=True)

        adapter = self.adapters[call.model_key]
        try:
            resp = _call_with_retry(adapter, call, self.max_tokens)
        except Exception as exc:  # retries exhausted
            resp = Response("", None, None, 0.0,
                            error=f"retries exhausted: {type(exc).__name__}: {exc}")
        self.cache.put(key, resp)
        parsed, outcome = score_outcome(resp, call.expected)
        return Result(call, resp, parsed, outcome)

    def run(self, calls: list[Call], progress=None) -> list[Result]:
        results: list[Result] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._run_one, c): c for c in calls}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                self._persist(res)
                if progress is not None:
                    progress(res)
        return results

    def _persist(self, res: Result) -> None:
        if self.responses_path is None:
            return
        with self._write_lock:
            with open(self.responses_path, "a") as fh:
                fh.write(json.dumps(res.row()) + "\n")


def write_manifest(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created": datetime.now(timezone.utc).isoformat(), **fields}
    path.write_text(json.dumps(payload, indent=2, default=str))


def git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
