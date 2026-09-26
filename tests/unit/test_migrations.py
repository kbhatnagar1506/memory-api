"""The migration history, the schema guard's decisions, and what ships with them.

No database here: the scripts are read as scripts and the guard's decision
table is driven through a store whose database calls are stubbed. The same
guard against a real Postgres is tests/conformance/test_schema_migrations.py.
"""

from __future__ import annotations

import importlib.util
import itertools
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from mapi.config import Environment, Settings, StoreBackend
from mapi.core.errors import ConfigurationError
from mapi.store import build_store
from mapi.store.postgres.store import SCHEMA_REVISION, PostgresStore, _schema_state

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "migrations" / "versions"


def _scripts() -> ScriptDirectory:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return ScriptDirectory.from_config(config)


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the history ------------------------------------------------------------------


def test_the_history_has_one_head_and_it_is_what_the_code_expects() -> None:
    """A second head makes `alembic upgrade head` refuse to run at all, and a
    head the store does not expect makes every staging boot refuse. Both are
    caught here instead of on a deploy."""
    assert _scripts().get_heads() == [SCHEMA_REVISION]


def test_the_history_is_one_line_of_zero_padded_numbers() -> None:
    """The guard orders revisions numerically (`_schema_state`), which is only
    sound while every revision is a 4-digit number and each follows the last."""
    revisions = list(reversed(list(_scripts().walk_revisions())))
    ids = [r.revision for r in revisions]
    assert all(i.isdigit() and len(i) == 4 for i in ids), ids
    assert ids == sorted(ids)
    for earlier, later in itertools.pairwise(revisions):
        assert later.down_revision == earlier.revision, (
            f"{later.revision} does not follow {earlier.revision}: the history forked"
        )
    assert ids[0] == "0001" and ids[-1] == SCHEMA_REVISION


def test_0007_chains_after_0006_whenever_0006_exists() -> None:
    """0006 is built on a parallel branch; 0007 must land after it, not beside."""
    module = _load(VERSIONS / "0007_drop_rls_bypass.py")
    has_0006 = any(VERSIONS.glob("0006_*.py"))
    assert module.down_revision == ("0006" if has_0006 else "0005")


def test_0007_drops_exactly_the_policies_0005_created() -> None:
    """Same tables, same policy names, and a downgrade that puts them back."""
    ours = _load(VERSIONS / "0007_drop_rls_bypass.py")
    theirs = _load(VERSIONS / "0005_row_level_security.py")
    assert ours.TENANT_TABLES == theirs.TENANT_TABLES

    executed: list[str] = []

    class _Op:
        @staticmethod
        def execute(sql: str) -> None:
            executed.append(" ".join(sql.split()))

    ours.op = _Op()
    ours.upgrade()
    assert executed == [
        f"DROP POLICY IF EXISTS {t}_admin_bypass ON {t}" for t in theirs.TENANT_TABLES
    ]
    executed.clear()
    ours.downgrade()
    assert len(executed) == len(theirs.TENANT_TABLES)
    assert all("_admin_bypass" in s and "app.bypass_rls" in s for s in executed)


# -- the guard's decision table ---------------------------------------------------


@pytest.mark.parametrize(
    ("revision", "state"),
    [
        (None, "absent"),
        (SCHEMA_REVISION, "current"),
        ("0005", "behind"),
        ("0001", "behind"),
        ("9999", "ahead"),
        ("0006,0007", "unknown"),
        ("abc1", "unknown"),
        ("12345", "unknown"),
    ],
)
def test_schema_state(revision: str | None, state: str) -> None:
    assert _schema_state(revision) == state


def _store(monkeypatch: pytest.MonkeyPatch, revision: str | None, *, required: bool):
    """A PostgresStore that never connects: revision stubbed, DDL recorded."""
    store = PostgresStore(
        "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none", require_migrated=required
    )
    calls: list[str] = []

    async def _revision() -> str | None:
        return revision

    async def _create() -> None:
        calls.append("ddl")

    monkeypatch.setattr(store, "schema_revision", _revision)
    monkeypatch.setattr(store, "_create_schema", _create)
    return store, calls


@pytest.mark.parametrize("required", [True, False])
async def test_a_current_schema_runs_no_ddl(monkeypatch, required: bool) -> None:
    store, calls = _store(monkeypatch, SCHEMA_REVISION, required=required)
    await store.initialize()
    assert calls == []


@pytest.mark.parametrize("revision", [None, "0005", "0006,0007", "junk"])
async def test_required_refuses_anything_but_current_or_ahead(monkeypatch, revision) -> None:
    store, calls = _store(monkeypatch, revision, required=True)
    with pytest.raises(ConfigurationError, match="alembic upgrade head"):
        await store.initialize()
    assert calls == [], "refusing to boot must not have run DDL first"


async def test_a_schema_ahead_of_the_build_boots_untouched(monkeypatch) -> None:
    """An image rollback after an additive migration must not be an outage."""
    store, calls = _store(monkeypatch, "9999", required=True)
    await store.initialize()
    assert calls == []


@pytest.mark.parametrize("revision", [None, "0005"])
async def test_development_keeps_create_on_boot(monkeypatch, revision) -> None:
    store, calls = _store(monkeypatch, revision, required=False)
    await store.initialize()
    assert calls == ["ddl"]


async def test_development_leaves_an_unplaceable_schema_alone(monkeypatch) -> None:
    store, calls = _store(monkeypatch, "0006,0007", required=False)
    await store.initialize()
    assert calls == []


@pytest.mark.parametrize(
    ("environment", "required"),
    [
        (Environment.LOCAL, False),
        (Environment.TEST, False),
        (Environment.STAGING, True),
        (Environment.PRODUCTION, True),
    ],
)
def test_staging_and_production_require_migrations(environment, required) -> None:
    store = build_store(
        Settings(
            environment=environment,
            store_backend=StoreBackend.POSTGRES,
            database_url="postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none",
        )
    )
    assert isinstance(store, PostgresStore)
    assert store._require_migrated is required


# -- what ships -------------------------------------------------------------------


def test_the_image_carries_its_migrations() -> None:
    """`alembic upgrade head` has to be runnable from the image that needs it."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY alembic.ini ./" in dockerfile
    assert "COPY migrations ./migrations" in dockerfile


def test_the_build_context_is_an_allowlist() -> None:
    """Everything the Dockerfile copies is let through; nothing else is."""
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    allowed = {line[1:].rstrip("/") for line in lines if line.startswith("!")}
    assert allowed == {"pyproject.toml", "README.md", "src", "alembic.ini", "migrations"}


def test_the_benchmark_judge_is_not_a_runtime_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith("anthropic") for d in project["dependencies"])
    assert any(d.startswith("anthropic") for d in project["optional-dependencies"]["bench"])
    requirements = (ROOT / "requirements.txt").read_text()
    assert "anthropic" not in requirements


def test_ci_runs_the_postgres_lane_as_production_would() -> None:
    """768 dimensions, a migrated database, a role RLS binds. Each of the three
    was missing and each one alone turned the lane red or vacuous."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'MAPI_TEST_DIMENSIONS: "768"' in workflow
    assert "alembic upgrade head" in workflow
    assert "NOSUPERUSER" in workflow and "NOBYPASSRLS" in workflow
