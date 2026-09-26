"""The runtime image can migrate the database it serves.

Row-level security lives only in the alembic history (0005); the app's own
create_all builds the tables without it. An image that cannot run
`alembic upgrade head` pushes every deploy into migrating from some other
checkout, which is how schema and code drift apart. So the runtime stage must
carry alembic.ini and migrations/ next to the venv that has alembic installed.
"""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _runtime_stage() -> list[str]:
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if re.match(r"FROM\s+\S+\s+AS\s+runtime", line)
    )
    stage = []
    for line in lines[start + 1 :]:
        if line.startswith("FROM "):
            break
        stage.append(line.strip())
    return stage


def _workdir(stage: list[str]) -> str:
    return [line.split(None, 1)[1] for line in stage if line.startswith("WORKDIR ")][-1]


def test_the_runtime_stage_copies_the_alembic_config_and_migrations() -> None:
    stage = _runtime_stage()
    copies = [
        line.split()[1:] for line in stage if line.startswith("COPY ") and "--from" not in line
    ]
    assert ["alembic.ini", "./"] in copies
    assert ["migrations", "./migrations"] in copies


def test_alembic_finds_the_migrations_from_the_image_workdir() -> None:
    """script_location is relative to the working directory alembic runs in."""
    ini = configparser.ConfigParser()
    ini.read(ROOT / "alembic.ini")
    assert ini["alembic"]["script_location"] == "migrations"
    assert _workdir(_runtime_stage()) == "/app"
    assert (ROOT / "migrations" / "env.py").is_file()
    assert sorted(p.name for p in (ROOT / "migrations" / "versions").glob("0005_*.py"))


def test_the_build_context_does_not_drop_the_migrations() -> None:
    """gcloud builds submit (no .gcloudignore) and docker (no .dockerignore)
    derive the upload from .gitignore; neither file may be excluded there."""
    ignored = [
        line.strip().strip("/")
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for path in ("alembic.ini", "migrations", "migrations/versions"):
        assert path not in ignored


def test_alembic_is_a_runtime_dependency() -> None:
    """The runtime venv installs only the project's dependencies plus [gemini]."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(re.match(r"alembic\b", dep) for dep in deps)
