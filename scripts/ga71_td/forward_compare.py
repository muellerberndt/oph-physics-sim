from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.gallium import compute_mdg_forward_comparison, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Write diagnostic mDG forward comparison for GA71.")
    parser.add_argument("--ratios", default=Path("data/gallium/public/ground_state_ratios.json"), type=Path)
    parser.add_argument("--witness", default=Path("data/gallium/benchmarks/mdg_witness.json"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compute_mdg_forward_comparison(ratios=load_json(args.ratios), witness=load_json(args.witness))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
