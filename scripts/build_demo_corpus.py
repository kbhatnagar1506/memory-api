"""Cut a small, readable demo corpus out of LongMemEval-S.

The full file is 277MB and takes ~20s just to parse, which is not something a
demo server should do on every boot. This runs once and writes a few hundred
KB that `scripts.demo_server` can load instantly.

What it selects, and why:

  * one corpus per question type, so the profile switcher covers every
    capability the benchmark tests rather than six variations of the easiest
    one. `knowledge-update` comes first because that is the type our
    supersession graph exists for;
  * every session the benchmark labels as evidence, plus a sample of others,
    so a profile shows a real history and not just the answer;
  * USER turns only. The assistant's replies are this system's own output,
    not facts about the person, and a memory store that ingests its own
    replies starts believing them. The benchmark path is untouched by this:
    `bench/datasets/longmemeval.py` still ingests whole sessions, which is
    the granularity LongMemEval labels evidence at.

Run:  .venv/bin/python -m scripts.build_demo_corpus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SOURCE = Path("bench/data/longmemeval_s.json")
OUT = Path(__file__).with_name(".demo_corpus.json")

#: One profile per capability. Ordered so the switcher opens on the type that
#: exercises the graph hardest.
WANTED = (
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
)

#: Non-evidence sessions to keep per profile. Enough that the evidence is not
#: trivially the whole history, small enough to stay readable and quick.
FILLER_SESSIONS = 10

#: Turns longer than this are truncated. A 4,000-character monologue is one
#: memory in name only, and it makes every node in the graph look identical.
MAX_TURN_CHARS = 600


def _sessions(instance: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one instance into session records, evidence marked."""
    ids = instance.get("haystack_session_ids") or []
    dates = instance.get("haystack_dates") or []
    evidence = {str(s) for s in (instance.get("answer_session_ids") or [])}

    out: list[dict[str, Any]] = []
    for index, turns in enumerate(instance.get("haystack_sessions") or []):
        sid = str(ids[index] if index < len(ids) else f"s{index}")
        user_turns = [
            (t.get("content") or "").strip()[:MAX_TURN_CHARS]
            for t in turns
            if t.get("role") == "user" and (t.get("content") or "").strip()
        ]
        if not user_turns:
            continue
        out.append(
            {
                "id": sid,
                "date": str(dates[index] if index < len(dates) else ""),
                "is_evidence": sid in evidence,
                "turns": user_turns,
            }
        )
    return out


def build() -> dict[str, Any]:
    raw = json.loads(SOURCE.read_text())

    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in raw:
        by_type.setdefault(str(item.get("question_type", "unknown")), []).append(item)

    profiles: list[dict[str, Any]] = []
    for qtype in WANTED:
        pool = by_type.get(qtype) or []
        # Prefer an instance with more than one evidence session: a single
        # evidence session cannot show a fact being revised across time.
        pick = max(
            pool[:40],
            key=lambda i: len(i.get("answer_session_ids") or []),
            default=None,
        )
        if pick is None:
            continue

        sessions = _sessions(pick)
        evidence = [s for s in sessions if s["is_evidence"]]
        filler = [s for s in sessions if not s["is_evidence"]][:FILLER_SESSIONS]
        kept = sorted(evidence + filler, key=lambda s: s["date"])

        profiles.append(
            {
                "question_type": qtype,
                "question_id": str(pick.get("question_id", "")),
                "question": str(pick.get("question", "")).strip(),
                "answer": str(pick.get("answer", "")).strip(),
                "question_date": str(pick.get("question_date", "")),
                "evidence_session_ids": [s["id"] for s in evidence],
                "sessions": kept,
                "sessions_total": len(sessions),
            }
        )

    return {"source": "LongMemEval-S", "profiles": profiles}


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found -- see bench/README for the download")
    data = build()
    OUT.write_text(json.dumps(data))
    turns = sum(len(s["turns"]) for p in data["profiles"] for s in p["sessions"])
    print(f"wrote {OUT} -- {len(data['profiles'])} profiles, {turns} user turns")
    for p in data["profiles"]:
        print(
            f"  {p['question_type']:<26} {len(p['sessions']):>2} sessions "
            f"({len(p['evidence_session_ids'])} evidence, of {p['sessions_total']} total)"
        )


if __name__ == "__main__":
    main()
