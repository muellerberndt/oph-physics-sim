from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.gallium import load_json, validate_certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a GA71_TD_SOURCE_CERTIFICATE JSON shape and promotion gates.")
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()
    result = validate_certificate(load_json(args.certificate))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["valid_json_shape"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
