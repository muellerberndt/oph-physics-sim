from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.gallium import build_no_target_leak_dag, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the GA71 no-target-leak DAG receipt.")
    parser.add_argument("--manifest", default=Path("data/gallium/evidence/ga71_td/manifest.template.json"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dag = build_no_target_leak_dag(load_json(args.manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "no_target_leak_pass": dag["no_target_leak_pass"]}, indent=2))
    return 0 if dag["no_target_leak_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
