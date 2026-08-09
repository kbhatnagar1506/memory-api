"""On-disk embedding cache, keyed by content.

The 500-corpus LongMemEval run spends 82 minutes embedding 373,687 chunks, and
every re-run repeated all of it because the store is in-memory and dies with
the process. That made iterating on the *answer* path — which is where two
thirds of our error lives — cost an hour and a half per experiment, so
experiments did not happen.

Keyed by content hash rather than by position, so it survives changing the
corpus subset, the question sample, or the chunking parameters: any chunk whose
text was embedded before is free, wherever it now appears. The model name and
dimension are part of the key, so switching either one correctly misses rather
than silently returning vectors from a different embedding space — a cache that
mixes spaces produces plausible, wrong rankings and no error.

Vectors are stored as raw float32 bytes: 768 dims is 3KB per chunk, so the full
LongMemEval corpus is about 1.1GB. JSON would be roughly five times that for no
benefit.
"""

from __future__ import annotations

import array
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path


def cache_key(model: str, dimensions: int, text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return f"{model}|{dimensions}|{digest}"


class DiskVectorCache:
    """Content-addressed vector store. Safe to delete at any time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  key TEXT PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  data BLOB NOT NULL)"
        )
        # Durability is worth nothing here — a lost write just re-embeds — and
        # WAL plus relaxed sync makes bulk inserts an order of magnitude faster.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if not keys:
            return out
        # Chunked IN clauses: SQLite caps variables per statement at 999.
        for start in range(0, len(keys), 900):
            window = keys[start : start + 900]
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT key, dim, data FROM vectors WHERE key IN ({placeholders})",
                window,
            ).fetchall()
            for key, dim, blob in rows:
                values = array.array("f")
                values.frombytes(blob)
                if len(values) != dim:  # corrupt row; treat as a miss
                    continue
                out[key] = list(values)
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, items: Iterable[tuple[str, list[float]]]) -> None:
        payload = [
            (key, len(vector), array.array("f", vector).tobytes()) for key, vector in items
        ]
        if not payload:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, dim, data) VALUES (?, ?, ?)", payload
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        total = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return {"rows": total, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        self._conn.close()


class DiskClaimCache:
    """Write-time extraction results, keyed by document content.

    Same reasoning as the vector cache and a stronger case for it: extraction
    is one model round-trip per document, and a LongMemEval ingest is tens of
    thousands of documents. Without this, every re-run of the ANSWER path —
    which is where the experiments actually are — would re-extract the entire
    corpus. Keyed by content plus model, so changing the extraction model
    correctly misses rather than silently reusing another model's decomposition.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS claims (  key TEXT PRIMARY KEY,  payload TEXT NOT NULL)"
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, text: str) -> str:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        return f"{model}|{digest}"

    def get_many(self, keys: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
        out: dict[str, list[tuple[str, str]]] = {}
        if not keys:
            return out
        for start in range(0, len(keys), 900):
            window = keys[start : start + 900]
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT key, payload FROM claims WHERE key IN ({placeholders})", window
            ).fetchall()
            for key, payload in rows:
                try:
                    out[key] = [(r["fact"], r["quote"]) for r in json.loads(payload)]
                except (ValueError, TypeError, KeyError):
                    continue  # corrupt row; treat as a miss
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, items: Iterable[tuple[str, list[tuple[str, str]]]]) -> None:
        payload = [
            (key, json.dumps([{"fact": f, "quote": q} for f, q in claims]))
            for key, claims in items
        ]
        if not payload:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO claims (key, payload) VALUES (?, ?)", payload
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        total = self._conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        return {"rows": total, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        self._conn.close()


__all__ = ["DiskClaimCache", "DiskVectorCache", "cache_key"]
