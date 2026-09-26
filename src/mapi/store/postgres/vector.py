"""pgvector's binary wire format, for SQLAlchemy over asyncpg.

pgvector's SQLAlchemy type speaks TEXT: every vector bound into a statement is
formatted as "[0.0123,-0.0456,...]" in Python and parsed by the server, and
every vector read back is a string Python splits and converts float by float.
At 768 dimensions that is ~8 KB of text per vector each way, against ~3 KB of
packed float32 -- and the parse is the slow half, because it runs in the
interpreter.

Binary needs two things to agree, and this module is both of them:

  * an asyncpg codec on each connection, so the driver packs and unpacks the
    `vector` type itself (`register_vector_codec`, run from the engine's
    connect hook);
  * a column type that hands the driver the raw list instead of pre-rendering
    it as text (`BinaryVector`), which would otherwise reach the binary
    encoder as a string it cannot use.

The switch lives on the DIALECT rather than on the type, because the type is
declared once at import time on a shared model while each engine decides for
itself. SQLAlchemy memoizes bind processors per dialect, so the flag must be
set before the engine compiles its first statement -- the store does it on
construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pgvector import Vector as PgVector
from pgvector.sqlalchemy import VECTOR

#: Attribute set on an engine's dialect to route vectors through the codec.
BINARY_VECTORS_FLAG = "mapi_binary_vectors"


def _encode(value: Any) -> bytes:
    """Driver-side encoder. Accepts what `BinaryVector` or a raw caller binds.

    A string arrives when something binds pgvector's TEXT form directly -- a
    raw `text()` statement, or pgvector's own SQLAlchemy type on a model this
    module does not know about. With the codec installed that would otherwise
    be a hard driver error, so it is parsed rather than refused.
    """
    if isinstance(value, PgVector):
        return value.to_binary()
    if isinstance(value, str):
        return PgVector(PgVector._from_text(value)).to_binary()
    return PgVector(list(value)).to_binary()


async def register_vector_codec(conn: Any) -> bool:
    """Install the binary `vector` codec on one asyncpg connection.

    Looks the type's schema up rather than assuming `public`: managed hosts
    often install extensions into their own schema, and a codec registered
    against the wrong one fails with "unknown type". Returns False when the
    extension does not exist yet -- a first boot, before `initialize()` has
    created it -- so the caller can recycle the connection once it does.
    """
    schema = await conn.fetchval(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace "
        "WHERE e.extname = 'vector'"
    )
    if schema is None:
        return False
    await conn.set_type_codec(
        "vector",
        schema=schema,
        encoder=_encode,
        decoder=PgVector.from_binary,
        format="binary",
    )
    return True


class BinaryVector(VECTOR):
    """pgvector's column type, minus the text rendering when the codec is on.

    Reads need no override: the parent's result processor already accepts a
    decoded `pgvector.Vector` as well as text, and returns a list either way.
    """

    cache_ok = True

    def bind_processor(self, dialect: Any) -> Callable[[Any], Any] | None:
        if not getattr(dialect, BINARY_VECTORS_FLAG, False):
            return super().bind_processor(dialect)  # type: ignore[no-any-return]

        def process(value: Any) -> Any:
            # Lists and pgvector's own type go through untouched; anything
            # else list-like (a tuple, an array) is normalized to what the
            # encoder's fast path takes.
            if value is None or isinstance(value, (list, PgVector)):
                return value
            return list(value)

        return process


__all__ = ["BINARY_VECTORS_FLAG", "BinaryVector", "register_vector_codec"]
