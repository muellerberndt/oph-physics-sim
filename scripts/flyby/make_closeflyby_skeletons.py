from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.flyby import write_closeflyby_public_certificates


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit fail-closed CloseFlyBy(F) public certificates.")
    parser.add_argument("--out-dir", default=Path("data/flyby/certificates"), type=Path)
    parser.add_argument("--public-rows", default=Path("data/flyby/public/flyby_public_rows.csv"), type=Path)
    parser.add_argument("--source-manifest", default=Path("data/flyby/public/flyby_source_manifest.yaml"), type=Path)
    parser.add_argument("--raw-root", default=Path("data/flyby/raw"), type=Path)
    args = parser.parse_args()
    result = write_closeflyby_public_certificates(
        args.out_dir,
        public_rows_path=args.public_rows,
        source_manifest_path=args.source_manifest,
        raw_root=args.raw_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
