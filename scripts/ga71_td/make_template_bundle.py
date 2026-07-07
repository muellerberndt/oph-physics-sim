from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.gallium import write_ga71_template_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit the fail-closed GA71 transition-density template bundle.")
    parser.add_argument("--out-dir", default=Path("data/gallium/receipts"), type=Path)
    args = parser.parse_args()
    result = write_ga71_template_bundle(args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
