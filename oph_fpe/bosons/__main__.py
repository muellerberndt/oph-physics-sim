from __future__ import annotations

import argparse
from pathlib import Path

from oph_fpe.bosons.pipeline import write_wzh_campaign_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed OPH W/Z/H numerical source-closure backend."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    write_wzh_campaign_bundle(args.config, args.out)
    print(args.out / "wzh_source_closure_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
