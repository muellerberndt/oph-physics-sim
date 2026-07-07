from __future__ import annotations

import argparse
import json
from pathlib import Path


def pending_main(tool_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"{tool_name} replay hook for CloseFlyBy(F).")
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--flyby-id", default=None)
    args = parser.parse_args()
    result = {
        "tool": tool_name,
        "status": "OD_REPLAY_IMPLEMENTATION_PENDING",
        "flyby_id": args.flyby_id,
        "nonclaim": "This hook is present so the certificate pipeline fails closed until real tracking replay is implemented.",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2
