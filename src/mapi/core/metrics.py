"""Prometheus metrics.

Deliberately few, and all low-cardinality. Every label value becomes a separate
time series, so putting a memory id or a raw path in a label is how a metrics
backend falls over. Paths are the route *template* ("/v1/spaces/{space_id}"),
never the resolved URL.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "mapi_http_requests_total",
    "HTTP requests processed.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "mapi_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
SEARCH_LATENCY = Histogram(
    "mapi_search_duration_seconds",
    "End-to-end search latency.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
SEARCH_STAGE_LATENCY = Histogram(
    "mapi_search_stage_duration_seconds",
    "Per-stage search latency.",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=REGISTRY,
)
EMBEDDINGS = Counter(
    "mapi_embeddings_total",
    "Texts embedded.",
    ["provider", "outcome"],
    registry=REGISTRY,
)
RERANKS = Counter(
    "mapi_reranks_total",
    "Rerank invocations.",
    ["backend", "outcome"],
    registry=REGISTRY,
)
INGESTED = Counter(
    "mapi_memories_ingested_total",
    "Memories written.",
    ["outcome"],
    registry=REGISTRY,
)
RATE_LIMITED = Counter(
    "mapi_rate_limited_total",
    "Requests rejected by the rate limiter.",
    registry=REGISTRY,
)
STORE_UP = Gauge(
    "mapi_store_up",
    "1 when the storage backend answered its last health probe.",
    registry=REGISTRY,
)

__all__ = [
    "EMBEDDINGS",
    "INGESTED",
    "RATE_LIMITED",
    "REGISTRY",
    "REQUESTS",
    "REQUEST_LATENCY",
    "RERANKS",
    "SEARCH_LATENCY",
    "SEARCH_STAGE_LATENCY",
    "STORE_UP",
]
