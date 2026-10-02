"""Application settings.

Everything is environment-driven with safe defaults, so the service boots with
zero configuration for local development and demands explicit values for the
things that must never be defaulted in production (signing secrets, database
URLs). `Settings.validate_production()` is the gate that enforces that
difference rather than trusting an operator to remember.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class StoreBackend(StrEnum):
    #: Reference implementation. Real algorithms, no external dependency; used
    #: for tests, CI and zero-infrastructure demos.
    MEMORY = "memory"
    #: Production backend: PostgreSQL + pgvector.
    POSTGRES = "postgres"


class EmbeddingBackend(StrEnum):
    #: Deterministic, offline, no credentials. Keeps CI green and tests fast.
    DETERMINISTIC = "deterministic"
    GEMINI = "gemini"
    OPENAI = "openai"


class RerankBackend(StrEnum):
    NONE = "none"
    #: Lexical-overlap cross-encoder approximation. Cheap, offline, decent.
    HEURISTIC = "heuristic"
    #: True LLM listwise reranking.
    LLM = "llm"


class SynthesisBackend(StrEnum):
    """Provider for the derive path's map/compose calls.

    NONE keeps the product free of read-path LLM calls (the default and the
    posture the benchmarks run in); the /derive endpoint then returns 503
    rather than silently degrading. Search and ingest never depend on this.
    """

    NONE = "none"
    GEMINI = "gemini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MAPI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- service ----------------------------------------------------------
    environment: Environment = Environment.LOCAL
    service_name: str = "mapi"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    debug_errors: bool = Field(
        default=False,
        description="Include exception detail in HTTP responses. Never in production.",
    )

    # -- storage ----------------------------------------------------------
    store_backend: StoreBackend = StoreBackend.MEMORY
    database_url: str | None = Field(
        default=None,
        description="postgresql+asyncpg://user:pass@host:5432/db",
    )
    db_pool_size: int = Field(default=10, ge=1, le=200)
    db_max_overflow: int = Field(default=10, ge=0, le=200)
    db_statement_timeout_ms: int = Field(default=15_000, ge=100)

    # -- embeddings -------------------------------------------------------
    embedding_backend: EmbeddingBackend = EmbeddingBackend.DETERMINISTIC
    embedding_model: str = "text-embedding-004"
    embedding_dimensions: int = Field(default=768, ge=8, le=4096)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_timeout_s: float = Field(default=20.0, gt=0)
    embedding_cache_size: int = Field(default=4096, ge=0)

    # -- reranking --------------------------------------------------------
    #: Default OFF on measured evidence. The heuristic reranker is lexical
    #: overlap, which is a poor signal on long documents: on LongMemEval
    #: (~12,000-character sessions, n=500) it cost 8 points of full recall
    #: (0.968 -> 0.888). It helped on LoCoMo's ~400-character turns (+0.022
    #: MRR), so it is worth enabling for short-document corpora — but the
    #: downside is 4x the upside, and a default should take the safe side of
    #: an asymmetric bet.
    rerank_backend: RerankBackend = RerankBackend.NONE
    synthesis_backend: SynthesisBackend = SynthesisBackend.NONE
    #: Model for derive map/compose calls. Flash: many small extraction calls.
    synthesis_model: str = "gemini-2.5-flash"
    #: Output allowance for a derive call. Not generous for its own sake:
    #: on a thinking model, reasoning tokens come out of this same budget,
    #: and a completion cut mid-JSON parses to nothing and is indistinguish-
    #: able from "there was nothing to extract".
    synthesis_max_output_tokens: int = Field(default=4096, ge=256, le=65_536)
    #: Reasoning allowance. 128 is the value our best benchmark numbers were
    #: produced with; raising it to 4096 on the answer path measured -34
    #: questions, so more thinking is not a free upgrade here.
    synthesis_thinking_budget: int = Field(default=128, ge=0, le=32_768)

    #: Write-path extraction, configured separately from derivation because
    #: it runs per DOCUMENT rather than per question -- hundreds of thousands
    #: of calls on a full ingest, where throughput is the binding constraint.
    extraction_model: str = "gemini-2.5-flash"
    extraction_max_output_tokens: int = Field(default=8192, ge=256, le=65_536)
    #: Extraction is closer to transcription than reasoning; a large thinking
    #: budget spends latency on every write to restate a passage.
    extraction_thinking_budget: int = Field(default=0, ge=0, le=32_768)
    #: Query understanding: one sentence in, one word out, on the read path.
    #: The smallest model on offer, because the task is reading comprehension
    #: of a single sentence and a bigger model would buy latency rather than
    #: accuracy. Fails open to the regex classifier, so this being wrong or
    #: unreachable costs ranking quality and never availability.
    understand_queries: bool = Field(
        default=True,
        description=(
            "Classify each search's question shape with a small model before "
            "retrieving. Needs synthesis_backend; inert without one."
        ),
    )
    understanding_model: str = "gemini-2.5-flash-lite"
    #: One label. The allowance is small enough that a model which starts
    #: writing prose gets cut off and falls back rather than billing for it.
    understanding_max_output_tokens: int = Field(default=16, ge=1, le=256)
    understanding_thinking_budget: int = Field(default=0, ge=0, le=1024)
    #: Hard ceiling on how long search will wait for the label. Past this the
    #: regex decides and the search proceeds -- a slow vendor makes search
    #: dumber, never slower.
    understanding_timeout_s: float = Field(default=2.0, gt=0, le=30)

    rerank_model: str = "gemini-2.5-flash"
    rerank_candidates: int = Field(default=32, ge=1, le=256)
    rerank_timeout_s: float = Field(default=12.0, gt=0)

    # -- retrieval defaults ----------------------------------------------
    default_limit: int = Field(default=10, ge=1, le=200)
    max_limit: int = Field(default=100, ge=1, le=1000)
    candidate_multiplier: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Candidates fetched per requested result before fusion.",
    )
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion damping.")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_per_source: int = Field(
        default=0,
        ge=0,
        description=(
            "Most results one source document may contribute to a top-k. 0 is off. "
            "Only bites once write-time extraction is on, where a document becomes "
            "~13 retrievable units and a few documents can otherwise consume the "
            "whole window -- measured as full_recall@k 0.968 -> 0.948 with MRR "
            "rising to 0.987, i.e. sharper retrieval and worse coverage."
        ),
    )
    route_by_kind: bool = Field(
        default=False,
        description=(
            "Split the retrieval window between EPISODIC and DERIVED memories by "
            "question shape. Off by default and inert without derived memories. "
            "Measured need: write-time extraction moved six capabilities and the "
            "sign matched the memory kind each needs, six for six -- claims answer "
            "'what is true', episodes answer 'what happened'."
        ),
    )
    #: Results a comprehensive question may return. "What is our entire
    #: infrastructure" is not asking which memory ranks highest -- it is
    #: asking what the territory contains, and a ranked top-10 answers a
    #: different question. Measured on 25 stored facts: 10 came back, every
    #: score inside a 2% band, with the database and the cache missing.
    coverage_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Window for questions asking for a complete set rather than a best "
            "match. Still bounded -- 'everything' has to fit in a context window."
        ),
    )
    #: Nearest existing memories compared against a new write when
    #: supersession or conflict detection is on. Bounded because every
    #: candidate costs an embedding across the wire and a cosine in Python;
    #: unbounded it becomes a table scan on every write.
    #:
    #: 64 rather than the 256 this replaced: the proposers' own similarity
    #: floor is 0.72, and on real corpora the count of memories above that
    #: floor is in the single digits. The extra rows were being fetched and
    #: discarded.
    consolidation_candidates: int = Field(default=64, ge=1, le=512)
    #: Turns of surrounding context handed to the answerer with each
    #: retrieved memory. 0 restores the old behaviour exactly.
    #:
    #: Measured need: retrieval delivers complete evidence for 97.2% of
    #: LongMemEval questions and accuracy is 0.8255, so ~17.5% are answered
    #: wrong while the evidence is already in context, against a 2.8%
    #: retrieval ceiling. A memory is one chat turn; "yeah, three of them"
    #: retrieves correctly and cannot be answered from.
    answer_neighbours: int = Field(default=2, ge=0, le=8)
    #: Embed each chunk with a one-line header of date, title, source and
    #: tags. The STORED text is never changed -- see embeddings/context.py.
    #:
    #: Unlike write-time extraction this adds no retrieval units, so the
    #: crowding that cost extraction 21-54 questions cannot occur: same chunk
    #: count, same window, only the vector moves.
    contextual_embedding: bool = Field(default=True)
    #: Ceiling on the assembled answer context. Padding every anchor with
    #: context dilutes the prompt, which is the mirror image of the failure
    #: hydration fixes.
    answer_budget_chars: int = Field(default=24_000, ge=1_000, le=200_000)
    half_life_days: float = Field(
        default=180.0,
        gt=0,
        description="Recency half-life. Older memories decay toward this schedule.",
    )

    # -- per-tenant quotas ------------------------------------------------
    # All zero = unlimited, which is the shipped default. Rate limiting caps
    # how FAST a tenant calls; these cap how MUCH they accumulate, and the
    # third one is the only thing standing between one enthusiastic customer
    # and an unbounded embedding bill.
    max_memories_per_org: int = Field(
        default=0, ge=0, description="Memories one organization may hold. 0 is unlimited."
    )
    max_bytes_per_org: int = Field(
        default=0, ge=0, description="Stored content bytes per organization. 0 is unlimited."
    )
    max_writes_per_day: int = Field(
        default=0, ge=0, description="Non-duplicate writes per org per day. 0 is unlimited."
    )

    # -- ingestion --------------------------------------------------------
    max_content_bytes: int = Field(default=1_000_000, ge=1)
    chunk_target_tokens: int = Field(default=320, ge=16, le=4096)
    chunk_overlap_tokens: int = Field(default=48, ge=0, le=1024)
    #: Code and JSON (a function, a recipe) stay one chunk up to this size, and past it
    #: split only between structural units (domain/chunking.py).
    chunk_structured_max_tokens: int = Field(default=1536, ge=128, le=8192)
    dedupe_threshold: float = Field(
        default=0.97,
        ge=0.0,
        le=1.0,
        description="Cosine similarity at or above which a write is a duplicate.",
    )
    supersede_min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence a supersession proposal must reach before `auto_supersede` "
            "applies it. The two errors are not symmetric: a false supersession "
            "flips a true memory to SUPERSEDED and default retrieval then hides it, "
            "so the information is gone from every answer with no error anywhere. A "
            "missed supersession leaves both memories visible and rankable, where "
            "recency still favours the newer one. Cheap mistake, expensive mistake."
        ),
    )

    # -- browser auth ------------------------------------------------------
    #: Google OAuth. Absent means the dashboard offers no sign-in and the API
    #: still works on bearer keys -- the two paths are independent.
    google_client_id: str | None = None
    google_client_secret: str | None = None
    #: Signs the session cookie. Rotating it logs everyone out, which is the
    #: correct emergency response and the reason it is separate from the API
    #: key pepper: revoking sessions must not invalidate every API key.
    session_secret: str | None = None
    #: Absolute origin this app is reached at, used to build the OAuth
    #: redirect. Cannot be inferred from the request: behind a proxy the Host
    #: header is attacker-controlled, and a wrong redirect_uri is how an auth
    #: code ends up somewhere else.
    public_base_url: str | None = None
    #: Days a browser session stays valid.
    session_max_age_days: int = Field(default=14, ge=1, le=90)

    # -- providers --------------------------------------------------------
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # -- security ---------------------------------------------------------
    api_key_pepper: str = Field(
        default="dev-insecure-pepper",
        description="Server-side pepper mixed into API key hashes.",
    )
    bootstrap_admin_key: str | None = Field(
        default=None,
        description="If set, an admin key with this literal value is seeded at boot.",
    )
    rate_limit_per_minute: int = Field(default=600, ge=1)
    rate_limit_burst: int = Field(default=120, ge=1)
    redis_url: str | None = None
    #: Production without Redis, said on purpose: a private deployment whose only callers
    #: are its owner's own services (behind Cloud Run IAM), where a per-instance limit is
    #: an acceptable backstop. Never for a deployment the public can reach.
    rate_limit_per_instance_ok: bool = False

    # -- request handling --------------------------------------------------
    max_request_bytes: int = Field(default=8_000_000, ge=1024)
    request_timeout_s: float = Field(default=30.0, gt=0)

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_fits(cls, v: int, info: ValidationInfo) -> int:
        target = info.data.get("chunk_target_tokens", 320)
        if v >= target:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be < chunk_target_tokens ({target}); "
                "equal or larger overlap makes chunking non-terminating"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _platform_env(cls, values: Any) -> Any:
        """Adopt the host platform's connection strings when ours are unset.

        Heroku attaches addons by setting DATABASE_URL and REDIS_URL; it has
        no idea about our MAPI_ prefix, and requiring an operator to copy them
        across by hand is a step that gets forgotten exactly once, in
        production, where the app then refuses to boot.

        The scheme rewrite is not cosmetic. Heroku still issues `postgres://`,
        which SQLAlchemy dropped support for, and asyncpg needs naming
        explicitly -- so the platform's own URL is unusable verbatim and the
        failure surfaces as an opaque dialect error at first connection.

        Runs BEFORE validation rather than after: a top-level after-validator
        cannot return a rebuilt model when the object is constructed through
        `__init__`, which is exactly how settings are loaded, and pydantic
        warns rather than raising -- so the adoption silently did nothing.
        """
        if not isinstance(values, dict):
            return values
        if not values.get("database_url"):
            raw = os.getenv("DATABASE_URL", "")
            if raw:
                for prefix in ("postgres://", "postgresql://"):
                    if raw.startswith(prefix):
                        raw = "postgresql+asyncpg://" + raw[len(prefix) :]
                        break
                values["database_url"] = raw
        if not values.get("redis_url"):
            managed = os.getenv("REDIS_URL", "")
            if managed:
                values["redis_url"] = managed
        return values

    @model_validator(mode="after")
    def _coherent(self) -> Settings:
        if self.max_limit < self.default_limit:
            raise ValueError("max_limit must be >= default_limit")
        if self.store_backend is StoreBackend.POSTGRES and not self.database_url:
            raise ValueError("store_backend=postgres requires database_url")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    def validate_production(self) -> list[str]:
        """Configuration that is acceptable locally but not in production.

        Returned rather than raised so the caller decides whether to refuse to
        boot or merely warn, and so every problem is reported at once instead of
        one per restart.
        """
        problems: list[str] = []
        if not self.is_production:
            return problems
        if self.api_key_pepper == "dev-insecure-pepper":
            problems.append("api_key_pepper is still the development default")
        if self.debug_errors:
            problems.append("debug_errors leaks exception detail to clients")
        if self.store_backend is not StoreBackend.POSTGRES:
            problems.append("store_backend must be postgres in production")
        if self.embedding_backend is EmbeddingBackend.DETERMINISTIC:
            problems.append("embedding_backend=deterministic produces non-semantic vectors")
        if self.bootstrap_admin_key:
            problems.append("bootstrap_admin_key must not be set in production")
        if not self.redis_url and not self.rate_limit_per_instance_ok:
            problems.append(
                "redis_url is unset: rate limiting would be per-process only "
                "(set rate_limit_per_instance_ok for a private, internal deployment)"
            )
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests mutate the environment; the cache must not outlive that."""
    get_settings.cache_clear()


def settings_from_env(**overrides: object) -> Settings:
    """Build settings ignoring the process cache. Used by tests and the CLI."""
    return Settings(**overrides)  # type: ignore[arg-type]


__all__ = [
    "EmbeddingBackend",
    "Environment",
    "RerankBackend",
    "Settings",
    "StoreBackend",
    "get_settings",
    "os",
    "reset_settings_cache",
    "settings_from_env",
]
