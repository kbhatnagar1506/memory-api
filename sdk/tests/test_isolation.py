"""The SDK must never import the server.

This is the property that makes the package publishable at all: shipping a
client that imports the engine would ship the engine. A test rather than a
convention, because the failure is silent -- someone reaches for one shared
helper, and the next release carries the retrieval pipeline, the prompts and
the benchmark harness onto PyPI.
"""

from __future__ import annotations

import ast
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "mapi_sdk"

#: Everything the client is allowed to depend on. httpx and the standard
#: library; each addition beyond this is a version conflict in somebody
#: else's application.
ALLOWED_THIRD_PARTY = {"httpx"}


def _imports(path: pathlib.Path) -> set[str]:
    """Top-level module names this file imports.

    Parsed, not grepped. A regex over the source also matches prose: a
    docstring line beginning "from it, since..." reads as an import of a
    module named `it`, and a test that cries wolf gets muted rather than
    fixed.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_no_module_imports_the_server_package() -> None:
    for path in SRC.rglob("*.py"):
        assert "mapi" not in {i for i in _imports(path) if i != "mapi_sdk"}, (
            f"{path.name} imports the server package"
        )


def test_only_httpx_and_the_standard_library() -> None:
    stdlib = set(sys.stdlib_module_names)
    for path in SRC.rglob("*.py"):
        for name in _imports(path):
            if name in stdlib or name.startswith("_") or name == "mapi_sdk":
                continue
            assert name in ALLOWED_THIRD_PARTY, f"{path.name} imports {name!r}"


def test_importing_the_sdk_does_not_pull_in_the_server() -> None:
    """Even transitively: the proof is the loaded module table."""
    import mapi_sdk  # noqa: F401

    leaked = [m for m in sys.modules if m == "mapi" or m.startswith("mapi.")]
    assert leaked == [], f"importing the SDK loaded server modules: {leaked}"


def test_no_server_source_is_packaged() -> None:
    """A wheel built from this tree must contain the client only."""
    names = {p.name for p in SRC.rglob("*.py")}
    assert names, "the package has no modules"
    for suspicious in ("service.py", "pipeline.py", "harness.py", "extract.py"):
        assert suspicious not in names, f"{suspicious} looks like server code"
