from __future__ import annotations

import argparse
import json
from pathlib import Path


def pending_main(tool_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"{tool_name} source-side hook for GA71_TD_SOURCE_CERTIFICATE.")
    parser.add_argument("--out", default=None, type=Path)
    args, _ = parser.parse_known_args()
    result = {
        "tool": tool_name,
        "status": "GA71_TD_CERTIFICATE_AWAITING_SOURCE_SIMULATION",
        "promotion_valid": False,
        "nonclaim": "This hook fails closed until a real source-side A=71 nuclear calculation is implemented.",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2
