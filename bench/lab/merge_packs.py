"""Merge authored persona packs into one corpus file, rejecting the broken ones.

Nine personas were authored by nine independent agents, and the failure modes
of that are mechanical: a missing kind, a duplicated week, an evidence quote
retyped rather than copied. `preference_v2.gate` already catches those PER
QUESTION at run time. This catches them PER PACK at merge time, so a pack that
would contribute nothing is never written into the corpus in the first place --
and so the corpus file on disk is a thing you can trust without re-deriving.

    python -m bench.lab.merge_packs out.json in1.json in2.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .preference_v2 import gate


def load(path: Path) -> list[dict]:
    """A file is either a bare pack or a `{"packs": [...]}` envelope."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "packs" in data:
        return list(data["packs"])
    return [data]


def main(out: Path, sources: list[Path]) -> int:
    packs: list[dict] = []
    seen: set[str] = set()
    for path in sources:
        try:
            candidates = load(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  SKIP {path.name}: unreadable ({exc})")
            continue
        for pack in candidates:
            name = str(pack.get("persona", path.stem))
            if name in seen:
                print(f"  SKIP {path.name}: duplicate persona {name[:40]!r}")
                continue
            # Gate the pack ALONE: a pack whose questions all drop is dead
            # weight that still costs 23 tagging calls and pads the run.
            usable, dropped = gate([pack])
            if len(usable) < 10:
                print(f"  SKIP {path.name}: only {len(usable)}/20 usable")
                for reason in dropped[:4]:
                    print(f"        {reason}")
                continue
            seen.add(name)
            packs.append(pack)
            kinds: dict[str, int] = {}
            for q in usable:
                kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
            spread = " ".join(f"{k[:4]}:{v}" for k, v in sorted(kinds.items()))
            print(f"  keep {path.name}: {len(usable):2}/20 usable  {spread}  {name[:44]}")

    usable, dropped = gate(packs)
    out.write_text(json.dumps({"packs": packs}, indent=1))
    print(f"\n{len(packs)} packs -> {len(usable)} usable questions ({len(dropped)} dropped)")
    kinds = {}
    for q in usable:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    print(f"by kind: {dict(sorted(kinds.items()))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]))
