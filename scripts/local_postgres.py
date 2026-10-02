"""A local Postgres with pgvector for the Postgres tests, set up the way CI does it.

    python3.12 -m venv .pg && .pg/bin/pip install pgserver     # Postgres + pgvector, no Docker
    .pg/bin/python scripts/local_postgres.py                    # prints the URL to export

Starts (or reuses) a server in .pgdata, creates the ordinary role CI connects as (no
superuser, no BYPASSRLS, so row-level security is in force) and the test database with
pgvector, then prints the MAPI_TEST_DATABASE_URL to run the suite with:

    MAPI_TEST_DATABASE_URL=... MAPI_TEST_DIMENSIONS=768 pytest -m "not slow"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(datadir: Path) -> None:
    import pgserver  # pip install pgserver (wheels for Python 3.9-3.12)

    server = pgserver.get_server(str(datadir), cleanup_mode=None)
    socket_dir = server.get_uri().split("host=")[-1]
    psql = Path(pgserver.__file__).parent / "pginstall" / "bin" / "psql"
    for database, statements in (
        (
            "postgres",
            [
                "DO $$ BEGIN CREATE ROLE mapi_app LOGIN PASSWORD 'mapi_app'"
                " NOSUPERUSER NOBYPASSRLS;"
                " EXCEPTION WHEN duplicate_object THEN NULL; END $$",
                "CREATE DATABASE mapi_test",  # an error when it exists already, which is fine
                "GRANT CREATE ON DATABASE mapi_test TO mapi_app",
            ],
        ),
        (
            "mapi_test",
            [
                "CREATE EXTENSION IF NOT EXISTS vector",
                "GRANT ALL ON SCHEMA public TO mapi_app",
            ],
        ),
    ):
        for sql in statements:
            os.spawnv(
                os.P_WAIT,
                str(psql),
                [
                    str(psql),
                    "-q",
                    "-h",
                    socket_dir,
                    "-U",
                    "postgres",
                    "-d",
                    database,
                    "-c",
                    sql,
                ],
            )
    print(
        f"export MAPI_TEST_DATABASE_URL='postgresql+asyncpg://mapi_app:mapi_app@/mapi_test?host={socket_dir}'"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".pgdata").resolve())
