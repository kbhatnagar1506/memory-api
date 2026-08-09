"""The replay demo is a claim we make, so it is tested like one.

It asserts something no extraction-based memory system can: that erasing a
source propagates into everything COMPUTED from it. A demo that silently
stops proving that is worse than no demo, because it still prints PASS.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from mapi.cli import main


def _run() -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["demo-replay"])
    assert code == 0, "demo-replay exited non-zero"
    return buffer.getvalue()


def test_demo_proves_erasure_reaches_live_past_and_history() -> None:
    """Delete removes the row; erase must also scrub the reconstructed past,
    or `?as_of=` resurrects content the caller was told is gone."""
    out = _run()
    assert "live read:            GONE" in out
    assert "point-in-time read:   GONE" in out
    assert "version history:      GONE" in out


def test_demo_proves_derivations_are_invalidated_not_left_standing() -> None:
    """The differentiating claim. A fact computed from erased evidence must
    stop being served -- extraction-based systems cannot do this because
    their derived facts have no precise lineage to walk."""
    out = _run()
    assert "status after erasing its source: STALE" in out
    assert "still returned by search: False" in out


def test_demo_keeps_serving_the_surviving_answer() -> None:
    """Erasure must be surgical. If the current policy stopped being served
    too, we would have proved destruction rather than propagation."""
    out = _run()
    assert "retained for 30 days" in out
    assert "PASS - erasure propagated through the graph" in out
