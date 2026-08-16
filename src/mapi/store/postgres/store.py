"""PostgreSQL + pgvector implementation of MemoryStore.

Held to the same conformance suite as the in-memory backend, which is what
guarantees the two agree on the semantics the retrieval layer depends on:
tenant scoping, filter behaviour, cursor stability and score direction.

Two details that are easy to get wrong and expensive to discover later:

  * pgvector's `<=>` is cosine *distance*. The retrieval layer works in cosine
    *similarity*, so every query converts with `1 - distance`. Returning the
    distance would silently invert the entire ranking.
  * Filters are applied inside the same statement as the ANN scan, not
    afterwards. Post-filtering a top-k means a query restricted to one tag can
    return nothing at all while matching rows sit just outside k.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ...core.errors import (
    BadRequestError,
    ConflictError,
    StoreError,
    ValidationError,
)
from ...core.ids import new_id
from ...core.logging import get_logger
from ...domain.embeddings.base import Vector
from ...domain.models import (
    ApiKey,
    Chunk,
    MemberRole,
    Membership,
    Memory,
    MemoryKind,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    Scope,
    Space,
    User,
    utcnow,
)
from ..base import (
    EraseReport,
    LexicalHit,
    MemoryFilter,
    MemoryStore,
    Page,
    TenantUsage,
    VectorHit,
)
from .models import (
    ApiKeyRow,
    Base,
    ChunkRow,
    MembershipRow,
    MemoryRow,
    MemoryVersionRow,
    OrganizationRow,
    RelationEdgeRow,
    SpaceRow,
    UserRow,
)

log = get_logger(__name__)

#: pgvector's own default for `hnsw.ef_search`, used as a floor so that sizing
#: this per query can only ever widen the search list, never narrow it.
_HNSW_EF_SEARCH_FLOOR = 40
#: pgvector's documented maximum for the parameter.
_HNSW_EF_SEARCH_CEILING = 1_000


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> str:
    try:
        padding = "=" * (-len(cursor) % 4)
        return base64.urlsafe_b64decode(cursor + padding).decode()
    except Exception as exc:
        raise BadRequestError("malformed cursor", field="cursor") from exc


def _require_aware(value: datetime) -> datetime:
    """Timezone-aware form of a value the domain guarantees is present."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _to_user(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        google_sub=row.google_sub,
        name=row.name,
        picture=row.picture,
        created_at=row.created_at,
        last_seen_at=row.last_seen_at,
    )


def _to_membership(row: MembershipRow) -> Membership:
    return Membership(
        id=row.id,
        user_id=row.user_id,
        org_id=row.org_id,
        role=MemberRole(row.role),
        created_at=row.created_at,
    )


class PostgresStore(MemoryStore):
    def __init__(
        self,
        database_url: str,
        *,
        dimensions: int = 768,
        pool_size: int = 10,
        max_overflow: int = 10,
        statement_timeout_ms: int = 15_000,
        echo: bool = False,
        max_scan_tuples: int = 20_000,
    ) -> None:
        self.dimensions = dimensions
        #: None = not probed yet; probed on first vector search.
        self._iterative_scan_supported: bool | None = None
        self._max_scan_tuples = max_scan_tuples
        self._engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=echo,
            connect_args={"server_settings": {"statement_timeout": str(statement_timeout_ms)}},
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    @staticmethod
    async def _scope(session: Any, org_id: str) -> None:
        """Bind this transaction to one tenant, for the database's benefit.

        Row-level security policies (migration 0005) compare `org_id` against
        the `app.org_id` setting. Unset, `current_setting(..., true)` is NULL,
        every comparison is NULL, and every row is filtered out -- so a store
        method that forgets this call sees an empty database rather than
        somebody else's.

        `set_config(..., true)` is TRANSACTION-local, which is the only safe
        scope here: sessions come from a connection pool, and a session-level
        setting would outlive the request and greet whichever tenant checked
        the connection out next.

        This does not replace the `WHERE org_id = ...` clauses in the queries
        below. Both state the same predicate on purpose -- the application
        filter is the behaviour, this is the backstop for the day someone
        writes a query that forgets it.
        """
        await session.execute(
            text("SELECT set_config('app.org_id', :org, true)"), {"org": org_id}
        )

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            from .models import HNSW_INDEX_DDL

            await conn.execute(text(HNSW_INDEX_DDL))

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def ping(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            log.warning("postgres_ping_failed", error=str(exc)[:200])
            return False

    # -- mapping ------------------------------------------------------------

    @staticmethod
    def _to_memory(row: MemoryRow) -> Memory:
        return Memory(
            id=row.id,
            org_id=row.org_id,
            space_id=row.space_id,
            content=row.content,
            summary=row.summary,
            metadata=dict(row.meta or {}),
            tags=list(row.tags or []),
            source=row.source,
            kind=MemoryKind(row.kind),
            status=MemoryStatus(row.status),
            occurred_at=row.occurred_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            content_sha256=row.content_sha256,
            version=row.version,
            chunks=[
                Chunk(
                    id=c.id,
                    memory_id=c.memory_id,
                    ordinal=c.ordinal,
                    text=c.text,
                    token_estimate=c.token_estimate,
                    embedding=list(c.embedding) if c.embedding is not None else None,
                )
                for c in row.chunks
            ],
        )

    async def _enable_iterative_scan(self, session: Any, wanted: int = 0) -> None:
        """Turn on pgvector 0.8's iterative index scans for this transaction.

        `wanted` is how many rows the caller is about to ask the index for, and
        it sizes `hnsw.ef_search`. That parameter was never set, so every query
        ran at pgvector's default of 40 -- while the retrieval pipeline asks for
        `limit * candidate_multiplier` candidates, which is 60 at the shipped
        defaults and up to 600 for a comprehensive question. Requesting 60 rows
        from a 40-row search list is asking the index for more than it looked
        at: pgvector's own guidance is that ef_search must be at least k, and
        recall degrades well before it drops below that.

        This is invisible in every benchmark number this repo has, because
        `bench/run.py` ingests into `InMemoryStore` and searches it by brute
        force. Exact search cannot express this failure, so `full_recall@k =
        0.968` is an upper bound on production rather than a measurement of it.

        Sized at twice the request and floored at pgvector's default, so a small
        query is never made worse; capped at 1000, which is the parameter's
        documented ceiling.

        This is a correctness setting wearing a performance costume. Our vector
        queries always carry filters (tenant, space, status): pre-0.8, HNSW
        collects candidates FIRST and filters afterwards, so a selective filter
        can silently return fewer rows than asked for — matching rows exist but
        sit just outside the scan. Iterative scans keep pulling from the index
        until the limit is satisfied. `relaxed_order` may yield slightly
        out-of-distance-order rows; we re-rank the over-fetched pool in Python,
        so ordering is restored before anyone sees it.

        Capability-probed once per store instance: on pgvector < 0.8 the SET
        fails, we log once and never retry. SET LOCAL scopes the setting to the
        enclosing transaction — no leakage through the connection pool.
        """
        if self._iterative_scan_supported is False:
            return
        try:
            await session.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
            await session.execute(
                text(f"SET LOCAL hnsw.max_scan_tuples = {int(self._max_scan_tuples)}")
            )
            if wanted > 0:
                ef = max(_HNSW_EF_SEARCH_FLOOR, min(_HNSW_EF_SEARCH_CEILING, wanted * 2))
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
            self._iterative_scan_supported = True
        except Exception as exc:
            self._iterative_scan_supported = False
            # A failed SET aborts the enclosing transaction; without a rollback
            # the actual search that follows would die on InFailedSQLTransaction.
            await session.rollback()
            log.warning(
                "pgvector_iterative_scan_unavailable",
                error=str(exc)[:160],
                hint="pgvector >= 0.8 required; filtered ANN may under-return",
            )

    def _apply_filters(self, stmt: Any, filters: MemoryFilter, model: Any = MemoryRow) -> Any:
        stmt = stmt.where(model.status.in_([s.value for s in filters.statuses]))
        if filters.kinds:
            stmt = stmt.where(model.kind.in_([k.value for k in filters.kinds]))
        if filters.tags:
            stmt = stmt.where(model.tags.contains(list(filters.tags)))
        for key, value in filters.metadata:
            # JSONB CONTAINMENT, not text comparison.
            #
            # This was `model.meta[key].astext == str(value)`, which compares
            # the JSON TEXT of the stored value against Python's `str()` of the
            # filter value. Those agree for strings and ints and for nothing
            # else, so `MemoryFilter.matches` -- documented one line above its
            # own definition as the semantics "SQL backends must reproduce
            # exactly" -- was reproduced for two types out of eight:
            #
            #     True         'true'      vs str(True)  == 'True'      no match
            #     None         SQL NULL    vs 'None'                    no match
            #     1.0          '1'         vs '1.0'                     no match
            #     {"b": 1}     '{"b": 1}'  vs "{'b': 1}"  (repr)        no match
            #     ["a"]        '["a"]'     vs "['a']"                   no match
            #
            # Reachable from the product: `POST /search` accepts `metadata` as
            # `dict[str, Any]` with no shallowness validator on the search body,
            # so a caller filtering on a boolean got zero results against
            # Postgres and the right ones in any in-memory dev run. Silent, and
            # invisible to the conformance suite, whose only metadata fixture is
            # `{"k": "v"}` -- one of the two types that happened to work.
            #
            # `@>` containment matches Python equality for every JSON type,
            # compares numbers as numbers, and uses the GIN index on `meta`.
            stmt = stmt.where(model.meta.contains({key: value}))
        if filters.occurred_after is not None:
            stmt = stmt.where(model.occurred_at >= _aware(filters.occurred_after))
        if filters.occurred_before is not None:
            stmt = stmt.where(model.occurred_at <= _aware(filters.occurred_before))
        if filters.source is not None:
            stmt = stmt.where(model.source == filters.source)
        return stmt

    # -- organizations & spaces --------------------------------------------

    # -- identity ----------------------------------------------------------

    async def upsert_user(self, user: User) -> User:
        async with self._session() as session, session.begin():
            row = (
                await session.execute(
                    select(UserRow).where(UserRow.google_sub == user.google_sub)
                )
            ).scalar_one_or_none()
            if row is None:
                row = UserRow(
                    id=user.id,
                    email=user.email,
                    google_sub=user.google_sub,
                    name=user.name,
                    picture=user.picture,
                    created_at=user.created_at,
                    last_seen_at=user.last_seen_at,
                )
                session.add(row)
            else:
                # The id stays put across logins: memberships point at it, and
                # minting a new one on every sign-in would orphan them.
                row.email = user.email
                row.name = user.name or row.name
                row.picture = user.picture or row.picture
                row.last_seen_at = user.last_seen_at
            await session.flush()
            return _to_user(row)

    async def get_user(self, user_id: str) -> User | None:
        async with self._session() as session:
            row = await session.get(UserRow, user_id)
            return _to_user(row) if row is not None else None

    async def get_user_by_google_sub(self, google_sub: str) -> User | None:
        async with self._session() as session:
            row = (
                await session.execute(select(UserRow).where(UserRow.google_sub == google_sub))
            ).scalar_one_or_none()
            return _to_user(row) if row is not None else None

    async def create_membership(self, membership: Membership) -> Membership:
        async with self._session() as session, session.begin():
            session.add(
                MembershipRow(
                    id=membership.id,
                    user_id=membership.user_id,
                    org_id=membership.org_id,
                    role=membership.role.value,
                    created_at=membership.created_at,
                )
            )
            from sqlalchemy.exc import IntegrityError

            try:
                await session.flush()
            except IntegrityError as exc:
                raise ConflictError("already a member of that organization") from exc
        return membership

    async def list_memberships(self, user_id: str) -> list[Membership]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(MembershipRow)
                    .where(MembershipRow.user_id == user_id)
                    .order_by(MembershipRow.created_at)
                )
            ).scalars()
            return [_to_membership(r) for r in rows]

    async def get_membership(self, user_id: str, org_id: str) -> Membership | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = (
                await session.execute(
                    select(MembershipRow).where(
                        MembershipRow.user_id == user_id,
                        MembershipRow.org_id == org_id,
                    )
                )
            ).scalar_one_or_none()
            return _to_membership(row) if row is not None else None

    async def create_organization(self, org: Organization) -> Organization:
        async with self._session() as session, session.begin():
            await self._scope(session, org.id)
            session.add(OrganizationRow(id=org.id, name=org.name, created_at=org.created_at))
        return org

    async def get_organization(self, org_id: str) -> Organization | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.get(OrganizationRow, org_id)
            if row is None:
                return None
            return Organization(id=row.id, name=row.name, created_at=row.created_at)

    async def create_space(self, space: Space) -> Space:
        from sqlalchemy.exc import IntegrityError

        try:
            async with self._session() as session, session.begin():
                await self._scope(session, space.org_id)
                session.add(
                    SpaceRow(
                        id=space.id,
                        org_id=space.org_id,
                        slug=space.slug,
                        name=space.name,
                        description=space.description,
                        meta=space.metadata,
                        created_at=space.created_at,
                        updated_at=space.updated_at,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(
                f"slug {space.slug!r} is already used in this organization"
            ) from exc
        return space

    @staticmethod
    def _to_space(row: SpaceRow) -> Space:
        return Space(
            id=row.id,
            org_id=row.org_id,
            slug=row.slug,
            name=row.name,
            description=row.description,
            metadata=dict(row.meta or {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def get_space(self, org_id: str, space_id: str) -> Space | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.scalar(
                select(SpaceRow).where(SpaceRow.id == space_id, SpaceRow.org_id == org_id)
            )
            return self._to_space(row) if row else None

    async def get_space_by_slug(self, org_id: str, slug: str) -> Space | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.scalar(
                select(SpaceRow).where(SpaceRow.org_id == org_id, SpaceRow.slug == slug)
            )
            return self._to_space(row) if row else None

    async def list_spaces(self, org_id: str) -> list[Space]:
        async with self._session() as session:
            await self._scope(session, org_id)
            rows = await session.scalars(
                select(SpaceRow).where(SpaceRow.org_id == org_id).order_by(SpaceRow.id)
            )
            return [self._to_space(r) for r in rows]

    async def delete_space(self, org_id: str, space_id: str) -> bool:
        async with self._session() as session, session.begin():
            await self._scope(session, org_id)
            result = await session.execute(
                delete(SpaceRow).where(SpaceRow.id == space_id, SpaceRow.org_id == org_id)
            )
            return bool(cast(CursorResult[Any], result).rowcount)

    # -- api keys ------------------------------------------------------------

    async def create_api_key(self, key: ApiKey) -> ApiKey:
        from sqlalchemy.exc import IntegrityError

        try:
            async with self._session() as session, session.begin():
                await self._scope(session, key.org_id)
                session.add(
                    ApiKeyRow(
                        id=key.id,
                        org_id=key.org_id,
                        name=key.name,
                        key_hash=key.key_hash,
                        prefix=key.prefix,
                        scopes=[s.value for s in key.scopes],
                        created_at=key.created_at,
                        expires_at=key.expires_at,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError("api key already exists") from exc
        return key

    @staticmethod
    def _to_key(row: ApiKeyRow) -> ApiKey:
        return ApiKey(
            id=row.id,
            org_id=row.org_id,
            name=row.name,
            key_hash=row.key_hash,
            prefix=row.prefix,
            scopes=frozenset(Scope(s) for s in row.scopes),
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            expires_at=row.expires_at,
            revoked_at=row.revoked_at,
        )

    async def get_api_key_by_hash(self, key_hash: str) -> ApiKey | None:
        async with self._session() as session:
            row = await session.scalar(select(ApiKeyRow).where(ApiKeyRow.key_hash == key_hash))
            return self._to_key(row) if row else None

    async def list_api_keys(self, org_id: str) -> list[ApiKey]:
        async with self._session() as session:
            await self._scope(session, org_id)
            rows = await session.scalars(
                select(ApiKeyRow).where(ApiKeyRow.org_id == org_id).order_by(ApiKeyRow.id)
            )
            return [self._to_key(r) for r in rows]

    async def revoke_api_key(self, org_id: str, key_id: str) -> bool:
        async with self._session() as session, session.begin():
            await self._scope(session, org_id)
            row = await session.scalar(
                select(ApiKeyRow).where(ApiKeyRow.id == key_id, ApiKeyRow.org_id == org_id)
            )
            if row is None or row.revoked_at is not None:
                return False
            row.revoked_at = datetime.now(UTC)
            return True

    async def touch_api_key(self, key_id: str, when: datetime) -> None:
        try:
            async with self._session() as session, session.begin():
                row = await session.get(ApiKeyRow, key_id)
                if row is not None:
                    row.last_used_at = when
        except Exception as exc:
            log.debug("touch_api_key_failed", error=str(exc)[:120])

    # -- memories ------------------------------------------------------------

    async def upsert_memory(self, memory: Memory, *, now: datetime | None = None) -> Memory:
        when = _require_aware(now or utcnow())
        async with self._session() as session, session.begin():
            await self._scope(session, memory.org_id)
            row = await session.get(MemoryRow, memory.id)
            if row is None:
                row = MemoryRow(id=memory.id)
                session.add(row)
            row.org_id = memory.org_id
            row.space_id = memory.space_id
            row.content = memory.content
            row.summary = memory.summary
            row.meta = memory.metadata
            row.tags = memory.tags
            row.source = memory.source
            row.kind = memory.kind.value
            row.status = memory.status.value
            row.occurred_at = _require_aware(memory.occurred_at)
            row.created_at = _require_aware(memory.created_at)
            row.updated_at = _require_aware(memory.updated_at)
            row.content_sha256 = memory.content_sha256
            row.version = memory.version

            # Bitemporal history, in the same transaction as the row write so
            # the live table and its audit trail cannot disagree.
            #
            # A write that does not bump `version` is an in-place correction,
            # not a new state, so it overwrites the open snapshot rather than
            # opening a second one. Otherwise a caller that re-saves without
            # bumping would either violate uq_versions_memory_version here or
            # accumulate duplicate snapshots — and the in-memory backend
            # applies the same rule, which is what keeps the two in agreement.
            open_version = await session.scalar(
                select(MemoryVersionRow).where(
                    MemoryVersionRow.memory_id == memory.id,
                    MemoryVersionRow.valid_to.is_(None),
                )
            )
            snapshot = MemoryVersion.snapshot(memory, valid_from=when)
            if open_version is not None and open_version.version == memory.version:
                open_version.content = snapshot.content
                open_version.summary = snapshot.summary
                open_version.meta = snapshot.metadata
                open_version.tags = snapshot.tags
                open_version.source = snapshot.source
                open_version.status = snapshot.status.value
                open_version.occurred_at = _require_aware(snapshot.occurred_at)
            else:
                if open_version is not None:
                    open_version.valid_to = when
                session.add(
                    MemoryVersionRow(
                        id=snapshot.id,
                        memory_id=snapshot.memory_id,
                        org_id=snapshot.org_id,
                        space_id=snapshot.space_id,
                        version=snapshot.version,
                        content=snapshot.content,
                        summary=snapshot.summary,
                        meta=snapshot.metadata,
                        tags=snapshot.tags,
                        source=snapshot.source,
                        kind=snapshot.kind.value,
                        status=snapshot.status.value,
                        occurred_at=_require_aware(snapshot.occurred_at),
                        valid_from=when,
                    )
                )

            # Chunks are replaced wholesale: a re-embedded memory has entirely
            # new vectors, and reconciling them individually is more code and
            # more ways to leave a stale vector behind.
            await session.execute(delete(ChunkRow).where(ChunkRow.memory_id == memory.id))
            for chunk in memory.chunks:
                if chunk.embedding is not None and len(chunk.embedding) != self.dimensions:
                    raise StoreError(
                        f"chunk {chunk.id} has {len(chunk.embedding)} dimensions, "
                        f"index expects {self.dimensions}"
                    )
                session.add(
                    ChunkRow(
                        id=chunk.id,
                        memory_id=memory.id,
                        org_id=memory.org_id,
                        space_id=memory.space_id,
                        ordinal=chunk.ordinal,
                        text=chunk.text,
                        token_estimate=chunk.token_estimate,
                        embedding=chunk.embedding,
                    )
                )
        return memory

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.scalar(
                select(MemoryRow).where(
                    MemoryRow.id == memory_id,
                    MemoryRow.org_id == org_id,
                    MemoryRow.space_id == space_id,
                )
            )
            return self._to_memory(row) if row else None

    async def get_memories(
        self, org_id: str, space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        async with self._session() as session:
            await self._scope(session, org_id)
            rows = await session.scalars(
                select(MemoryRow).where(
                    MemoryRow.id.in_(list(memory_ids)),
                    MemoryRow.org_id == org_id,
                    MemoryRow.space_id == space_id,
                )
            )
            return {r.id: self._to_memory(r) for r in rows}

    async def _bridge_supersession(
        self, session: Any, org_id: str, space_id: str, memory_id: str
    ) -> int:
        """Preserve revision chains across the removal of `memory_id`.

        A supersedes B supersedes C: removing B severs the only path from A to
        C, and C — a fact the user explicitly replaced — resurfaces as current
        (found live: erasing the middle of a Portland -> Austin -> Seattle
        chain brought Portland back). For each (X -> memory, memory -> Y) pair
        of SUPERSEDES edges, insert X -> Y before the FK cascade or explicit
        delete takes the originals.

        Runs inside the caller's transaction so a failed removal cannot leave
        bridges to a memory that still exists. `ON CONFLICT DO NOTHING` against
        the (space, source, target, type) unique constraint makes it idempotent
        and race-safe. Only SUPERSEDES composes transitively; the other edge
        types are deliberately left to break visibly.

        The bridge references only the two surviving ids — nothing of the
        removed memory's content — so the erasure path stays compliant.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        incoming = (
            await session.execute(
                select(RelationEdgeRow.source_id, RelationEdgeRow.confidence).where(
                    RelationEdgeRow.org_id == org_id,
                    RelationEdgeRow.space_id == space_id,
                    RelationEdgeRow.target_id == memory_id,
                    RelationEdgeRow.type == RelationType.SUPERSEDES.value,
                )
            )
        ).all()
        outgoing = (
            await session.execute(
                select(RelationEdgeRow.target_id, RelationEdgeRow.confidence).where(
                    RelationEdgeRow.org_id == org_id,
                    RelationEdgeRow.space_id == space_id,
                    RelationEdgeRow.source_id == memory_id,
                    RelationEdgeRow.type == RelationType.SUPERSEDES.value,
                )
            )
        ).all()
        if not incoming or not outgoing:
            return 0
        rows = [
            {
                # "edge", not "rel". `new_id` validates its kind against
                # PREFIXES and raises, so this line made every supersession
                # bridge throw ValueError -- which is to say, deleting or
                # erasing any memory in the middle of a revision chain was a
                # 500, and the chain-preservation this function exists to do
                # never happened on Postgres. The in-memory store bridges
                # correctly, so nothing caught it until the conformance suite
                # ran against a real database for the first time.
                "id": new_id("edge"),
                "org_id": org_id,
                "space_id": space_id,
                "source_id": upstream.source_id,
                "target_id": downstream.target_id,
                "type": RelationType.SUPERSEDES.value,
                "reason": "bridged across a removed intermediate revision",
                "confidence": min(upstream.confidence, downstream.confidence),
            }
            for upstream in incoming
            for downstream in outgoing
            if upstream.source_id != downstream.target_id
        ]
        if not rows:
            return 0
        statement = (
            pg_insert(RelationEdgeRow)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_edges_triple")
        )
        result = await session.execute(statement)
        return int(cast(CursorResult[Any], result).rowcount or 0)

    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> bool:
        async with self._session() as session, session.begin():
            await self._scope(session, org_id)
            # Bridge BEFORE the row delete: the FK cascade on relation_edges
            # destroys this memory's edges in the same statement as the row.
            await self._bridge_supersession(session, org_id, space_id, memory_id)
            result = await session.execute(
                delete(MemoryRow).where(
                    MemoryRow.id == memory_id,
                    MemoryRow.org_id == org_id,
                    MemoryRow.space_id == space_id,
                )
            )
            return bool(cast(CursorResult[Any], result).rowcount)

    async def erase_memory(self, org_id: str, space_id: str, memory_id: str) -> EraseReport:
        # One transaction for the whole purge: an erasure that half-applies
        # leaves the system claiming content is gone while as_of still serves
        # it, which is worse than failing outright.
        async with self._session() as session, session.begin():
            await self._scope(session, org_id)
            chunk_count = (
                await session.scalar(
                    select(func.count())
                    .select_from(ChunkRow)
                    .where(
                        ChunkRow.memory_id == memory_id,
                        ChunkRow.org_id == org_id,
                        ChunkRow.space_id == space_id,
                    )
                )
                or 0
            )
            edges_bridged = await self._bridge_supersession(
                session, org_id, space_id, memory_id
            )
            edge_result = await session.execute(
                delete(RelationEdgeRow).where(
                    RelationEdgeRow.org_id == org_id,
                    RelationEdgeRow.space_id == space_id,
                    (RelationEdgeRow.source_id == memory_id)
                    | (RelationEdgeRow.target_id == memory_id),
                )
            )
            edges_removed = int(cast(CursorResult[Any], edge_result).rowcount or 0)

            version_result = await session.execute(
                delete(MemoryVersionRow).where(
                    MemoryVersionRow.memory_id == memory_id,
                    MemoryVersionRow.org_id == org_id,
                    MemoryVersionRow.space_id == space_id,
                )
            )
            versions_purged = int(cast(CursorResult[Any], version_result).rowcount or 0)

            row_result = await session.execute(
                delete(MemoryRow).where(
                    MemoryRow.id == memory_id,
                    MemoryRow.org_id == org_id,
                    MemoryRow.space_id == space_id,
                )
            )
            row_existed = bool(cast(CursorResult[Any], row_result).rowcount)

            if not row_existed and versions_purged == 0:
                return EraseReport(existed=False)
            return EraseReport(
                existed=True,
                chunks_removed=chunk_count if row_existed else 0,
                edges_removed=edges_removed,
                versions_purged=versions_purged,
                edges_bridged=edges_bridged,
            )

    async def list_memories(
        self,
        org_id: str,
        space_id: str,
        *,
        filters: MemoryFilter,
        limit: int,
        cursor: str | None = None,
    ) -> Page:
        if limit <= 0:
            return Page(items=[])
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = select(MemoryRow).where(
                MemoryRow.org_id == org_id, MemoryRow.space_id == space_id
            )
            stmt = self._apply_filters(stmt, filters)
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = await session.scalar(count_stmt) or 0

            if cursor:
                stmt = stmt.where(MemoryRow.id > _decode_cursor(cursor))
            stmt = stmt.order_by(MemoryRow.id).limit(limit + 1)
            rows = list(await session.scalars(stmt))

            has_more = len(rows) > limit
            window = rows[:limit]
            return Page(
                items=[self._to_memory(r) for r in window],
                next_cursor=_encode_cursor(window[-1].id) if window and has_more else None,
                total=total,
            )

    async def count_memories(self, org_id: str, space_id: str, *, filters: MemoryFilter) -> int:
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = select(MemoryRow.id).where(
                MemoryRow.org_id == org_id, MemoryRow.space_id == space_id
            )
            stmt = self._apply_filters(stmt, filters)
            return await session.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None:
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.scalar(
                select(MemoryRow)
                .where(
                    MemoryRow.org_id == org_id,
                    MemoryRow.space_id == space_id,
                    MemoryRow.content_sha256 == digest,
                    MemoryRow.status != MemoryStatus.ARCHIVED.value,
                )
                .order_by(MemoryRow.id)
                .limit(1)
            )
            return self._to_memory(row) if row else None

    # -- relations: a typed, indexed graph ------------------------------------

    @staticmethod
    def _to_edge(row: RelationEdgeRow) -> RelationEdge:
        return RelationEdge(
            id=row.id,
            org_id=row.org_id,
            space_id=row.space_id,
            source_id=row.source_id,
            target_id=row.target_id,
            type=RelationType(row.type),
            reason=row.reason,
            confidence=row.confidence,
            created_at=row.created_at,
        )

    async def create_relation(self, edge: RelationEdge) -> RelationEdge:
        async with self._session() as session, session.begin():
            await self._scope(session, edge.org_id)
            existing = await session.scalar(
                select(RelationEdgeRow).where(
                    RelationEdgeRow.space_id == edge.space_id,
                    RelationEdgeRow.source_id == edge.source_id,
                    RelationEdgeRow.target_id == edge.target_id,
                    RelationEdgeRow.type == edge.type.value,
                )
            )
            if existing is not None:
                return self._to_edge(existing)
            session.add(
                RelationEdgeRow(
                    id=edge.id,
                    org_id=edge.org_id,
                    space_id=edge.space_id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    type=edge.type.value,
                    reason=edge.reason,
                    confidence=edge.confidence,
                    created_at=_require_aware(edge.created_at),
                )
            )
        return edge

    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        if direction not in ("out", "in"):
            # ValidationError (422), not a bare ValueError. Unreachable over HTTP
            # today -- the route constrains it with Query(pattern="^(out|in)$") --
            # but a non-MapiError escaping the store becomes an opaque 500 through
            # the catch-all handler, so any future caller that forwards an
            # unvalidated direction would get "an unexpected error occurred"
            # instead of a problem document naming the field.
            raise ValidationError(
                f"direction must be 'out' or 'in', got {direction!r}", field="direction"
            )
        column = RelationEdgeRow.source_id if direction == "out" else RelationEdgeRow.target_id
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = select(RelationEdgeRow).where(
                RelationEdgeRow.org_id == org_id,
                RelationEdgeRow.space_id == space_id,
                column == memory_id,
            )
            if type is not None:
                stmt = stmt.where(RelationEdgeRow.type == type.value)
            # Oldest first, matching the in-memory backend: the walker relies
            # on this to follow the original supersession decision.
            stmt = stmt.order_by(RelationEdgeRow.created_at, RelationEdgeRow.id)
            return [self._to_edge(r) for r in await session.scalars(stmt)]

    async def get_relations_between(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        ids = list(memory_ids)
        if not ids:
            return []
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = select(RelationEdgeRow).where(
                RelationEdgeRow.org_id == org_id,
                RelationEdgeRow.space_id == space_id,
                RelationEdgeRow.source_id.in_(ids),
                RelationEdgeRow.target_id.in_(ids),
            )
            if type is not None:
                stmt = stmt.where(RelationEdgeRow.type == type.value)
            stmt = stmt.order_by(RelationEdgeRow.created_at, RelationEdgeRow.id)
            return [self._to_edge(r) for r in await session.scalars(stmt)]

    # -- bitemporal history ----------------------------------------------------

    @staticmethod
    def _to_version(row: MemoryVersionRow) -> MemoryVersion:
        return MemoryVersion(
            id=row.id,
            memory_id=row.memory_id,
            org_id=row.org_id,
            space_id=row.space_id,
            version=row.version,
            content=row.content,
            summary=row.summary,
            metadata=dict(row.meta or {}),
            tags=list(row.tags or []),
            source=row.source,
            kind=MemoryKind(row.kind),
            status=MemoryStatus(row.status),
            occurred_at=row.occurred_at,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
        )

    async def get_memory_as_of(
        self, org_id: str, space_id: str, memory_id: str, as_of: datetime
    ) -> Memory | None:
        moment = _require_aware(as_of)
        async with self._session() as session:
            await self._scope(session, org_id)
            row = await session.scalar(
                select(MemoryVersionRow)
                .where(
                    MemoryVersionRow.org_id == org_id,
                    MemoryVersionRow.space_id == space_id,
                    MemoryVersionRow.memory_id == memory_id,
                    MemoryVersionRow.valid_from <= moment,
                    # Half-open interval [valid_from, valid_to): a snapshot
                    # that closed exactly at `as_of` was already superseded.
                    (MemoryVersionRow.valid_to.is_(None))
                    | (MemoryVersionRow.valid_to > moment),
                )
                # Newest match, explicitly. Without this the query had no
                # ORDER BY and no LIMIT, so `scalar()` returned whichever row
                # the planner produced first, while the in-memory store scans
                # `reversed(versions)` and returns the newest. Unobservable
                # today -- the write path keeps validity intervals disjoint, so
                # at most one row can match -- but "correct because nothing has
                # broken the invariant yet" is not the same as correct, and any
                # backfill, import or manual insert that overlaps two intervals
                # would silently return an arbitrary version.
                .order_by(MemoryVersionRow.version.desc())
                .limit(1)
            )
            if row is None:
                return None
            version = self._to_version(row)
            return Memory(
                id=version.memory_id,
                org_id=version.org_id,
                space_id=version.space_id,
                content=version.content,
                summary=version.summary,
                metadata=version.metadata,
                tags=version.tags,
                source=version.source,
                kind=version.kind,
                status=version.status,
                occurred_at=version.occurred_at,
                version=version.version,
                chunks=[],
            )

    async def list_memory_versions(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[MemoryVersion]:
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = (
                select(MemoryVersionRow)
                .where(
                    MemoryVersionRow.org_id == org_id,
                    MemoryVersionRow.space_id == space_id,
                    MemoryVersionRow.memory_id == memory_id,
                )
                .order_by(MemoryVersionRow.valid_from, MemoryVersionRow.version)
            )
            return [self._to_version(r) for r in await session.scalars(stmt)]

    # -- retrieval ------------------------------------------------------------

    async def vector_search(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[VectorHit]:
        if not embedding or limit <= 0:
            return []
        if len(embedding) != self.dimensions:
            raise StoreError(
                f"query embedding has {len(embedding)} dimensions, "
                f"index expects {self.dimensions}"
            )
        async with self._session() as session:
            await self._scope(session, org_id)
            await self._enable_iterative_scan(session, wanted=limit)
            # `<=>` is cosine DISTANCE; the retrieval layer works in similarity.
            distance = ChunkRow.embedding.cosine_distance(embedding).label("distance")
            stmt = (
                select(ChunkRow.memory_id, ChunkRow.id, ChunkRow.text, distance)
                .join(MemoryRow, MemoryRow.id == ChunkRow.memory_id)
                .where(
                    ChunkRow.org_id == org_id,
                    ChunkRow.space_id == space_id,
                    ChunkRow.embedding.is_not(None),
                )
            )
            # Filters ride inside the ANN statement. Filtering after a top-k
            # scan would let a restrictive filter return nothing.
            stmt = self._apply_filters(stmt, filters)
            # Over-fetch: one memory can contribute several chunks, and we
            # return the best chunk per memory.
            stmt = stmt.order_by(distance).limit(limit * 4)

            best: dict[str, VectorHit] = {}
            for memory_id, chunk_id, chunk_text, dist in await session.execute(stmt):
                similarity = 1.0 - float(dist)
                current = best.get(memory_id)
                if current is None or similarity > current.score:
                    best[memory_id] = VectorHit(memory_id, chunk_id, similarity, chunk_text)
            hits = sorted(best.values(), key=lambda h: (-h.score, h.memory_id))
            return hits[:limit]

    async def tenant_usage(self, org_id: str, *, since: datetime) -> TenantUsage:
        """One aggregate over the org's memories, on the tenant index.

        `octet_length` on the stored content rather than a Python len(): the
        point of the byte limit is what is on disk, and reading every row
        back to measure it would make the quota check cost more than the
        write it guards.
        """
        async with self._session() as session:
            await self._scope(session, org_id)
            row = (
                await session.execute(
                    select(
                        func.count(MemoryRow.id),
                        func.coalesce(func.sum(func.octet_length(MemoryRow.content)), 0),
                        func.count(MemoryRow.id).filter(MemoryRow.created_at >= since),
                    ).where(
                        MemoryRow.org_id == org_id,
                        MemoryRow.status != MemoryStatus.ARCHIVED.value,
                    )
                )
            ).one()
            return TenantUsage(
                memories=int(row[0]), bytes_stored=int(row[1]), writes_today=int(row[2])
            )

    async def neighbours(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        exclude_id: str = "",
    ) -> list[tuple[Memory, Vector]]:
        """Nearest memories with the chunk embedding that matched.

        One statement, ordered by the HNSW index. The chunk embedding comes
        back with the row because the consolidation functions compute their
        own similarity -- see the interface docstring for why that is worth
        the extra column rather than trusting a score from here.
        """
        if not embedding or limit <= 0:
            return []
        if len(embedding) != self.dimensions:
            raise StoreError(
                f"query embedding has {len(embedding)} dimensions, "
                f"index expects {self.dimensions}"
            )
        async with self._session() as session:
            await self._scope(session, org_id)
            await self._enable_iterative_scan(session, wanted=limit)
            distance = ChunkRow.embedding.cosine_distance(embedding).label("distance")
            stmt = (
                select(MemoryRow, ChunkRow.embedding, distance)
                .join(ChunkRow, ChunkRow.memory_id == MemoryRow.id)
                .where(
                    ChunkRow.org_id == org_id,
                    ChunkRow.space_id == space_id,
                    ChunkRow.embedding.is_not(None),
                    MemoryRow.status != MemoryStatus.ARCHIVED.value,
                )
                # Over-fetch, because one memory can own several chunks and
                # only its best one should count toward the limit.
                .order_by(distance)
                .limit(limit * 4)
            )
            if exclude_id:
                stmt = stmt.where(MemoryRow.id != exclude_id)

            best: dict[str, tuple[float, Memory, Vector]] = {}
            for row, chunk_embedding, dist in await session.execute(stmt):
                similarity = 1.0 - float(dist)
                current = best.get(row.id)
                if current is None or similarity > current[0]:
                    best[row.id] = (
                        similarity,
                        self._to_memory(row),
                        list(chunk_embedding),
                    )
            ordered = sorted(best.values(), key=lambda t: (-t[0], t[1].id))
            return [(memory, vector) for _, memory, vector in ordered[:limit]]

    async def lexical_search(
        self,
        org_id: str,
        space_id: str,
        query: str,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[LexicalHit]:
        if not query.strip() or limit <= 0:
            return []
        async with self._session() as session:
            await self._scope(session, org_id)
            # websearch_to_tsquery tolerates arbitrary user input; plainto_ and
            # to_tsquery raise on characters a user will absolutely type.
            tsquery = func.websearch_to_tsquery("english", query)
            rank = func.ts_rank_cd(ChunkRow.search_vector, tsquery).label("rank")
            stmt = (
                select(ChunkRow.memory_id, ChunkRow.id, ChunkRow.text, rank)
                .join(MemoryRow, MemoryRow.id == ChunkRow.memory_id)
                .where(
                    ChunkRow.org_id == org_id,
                    ChunkRow.space_id == space_id,
                    ChunkRow.search_vector.op("@@")(tsquery),
                )
            )
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(rank.desc()).limit(limit * 4)

            best: dict[str, LexicalHit] = {}
            for memory_id, chunk_id, chunk_text, score in await session.execute(stmt):
                current = best.get(memory_id)
                if current is None or float(score) > current.score:
                    best[memory_id] = LexicalHit(memory_id, chunk_id, float(score), chunk_text)
            hits = sorted(best.values(), key=lambda h: (-h.score, h.memory_id))
            return hits[:limit]

    async def sample_embeddings(
        self, org_id: str, space_id: str, *, limit: int
    ) -> list[tuple[str, Vector]]:
        async with self._session() as session:
            await self._scope(session, org_id)
            stmt = (
                select(ChunkRow.memory_id, ChunkRow.embedding)
                .join(MemoryRow, MemoryRow.id == ChunkRow.memory_id)
                .where(
                    ChunkRow.org_id == org_id,
                    ChunkRow.space_id == space_id,
                    ChunkRow.ordinal == 0,
                    ChunkRow.embedding.is_not(None),
                    MemoryRow.status != MemoryStatus.ARCHIVED.value,
                )
                .order_by(ChunkRow.memory_id)
                .limit(limit)
            )
            return [
                (memory_id, list(vector)) for memory_id, vector in await session.execute(stmt)
            ]


__all__ = ["PostgresStore"]
