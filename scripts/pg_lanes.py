"""Run the Postgres lanes against a real database, isolated in one schema.

    MAPI_TEST_DATABASE_URL=postgresql+asyncpg://role:...@host/db \\
        .venv/bin/python -m scripts.pg_lanes --schema ws3 [--drop] [-- pytest args]

The conformance, row-level-security and migration lanes need a MIGRATED
database, and a shared one (one test database, several people) needs each
run kept apart. This does both:

  1. CREATE SCHEMA IF NOT EXISTS <schema>
  2. `alembic upgrade head` inside it, as the connecting role -- which should
     be shaped like production's: owns what it creates, is neither superuser
     nor BYPASSRLS, so the RLS lane has something to test
  3. pytest tests/conformance -m postgres, with MAPI_TEST_SCHEMA pinning every
     connection to that schema and MAPI_TEST_DIMENSIONS at 768 unless set
  4. with --drop, DROP SCHEMA ... CASCADE afterwards, pass or fail

The URL is never printed: it carries the password.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

from tests.support import postgres as pg_support


async def _prepare(schema: str) -> None:
    admin = pg_support.make_store(in_schema=schema)
    try:
        async with admin._engine.begin() as conn:
            from sqlalchemy import text

            # The pinned search_path does not need the schema to exist yet.
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        await pg_support.migrate(admin)
        print(f"schema {schema}: migrated to {await admin.schema_revision()}")
    finally:
        await admin.aclose()


async def _drop(schema: str) -> None:
    admin = pg_support.make_store(in_schema="public")
    try:
        async with admin._engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        print(f"schema {schema}: dropped")
    finally:
        await admin.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pg_lanes", description=__doc__)
    parser.add_argument("--schema", required=True, help="lowercase identifier")
    parser.add_argument("--drop", action="store_true", help="drop the schema afterwards")
    parser.add_argument("pytest_args", nargs="*", help="extra arguments for pytest")
    args = parser.parse_args(argv)

    if not pg_support.database_url():
        print("MAPI_TEST_DATABASE_URL is not set", file=sys.stderr)
        return 2
    os.environ["MAPI_TEST_SCHEMA"] = args.schema
    pg_support.schema()  # validates the identifier before anything runs
    os.environ.setdefault("MAPI_TEST_DIMENSIONS", "768")

    asyncio.run(_prepare(args.schema))
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/conformance",
                "-m",
                "postgres",
                "-p",
                "no:cacheprovider",
                *args.pytest_args,
            ]
        )
    finally:
        if args.drop:
            asyncio.run(_drop(args.schema))


if __name__ == "__main__":
    sys.exit(main())
