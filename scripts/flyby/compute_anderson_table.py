from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from oph_fpe.flyby import anderson_mm_s, write_anderson_summary

K_MM_PER_KMS = 3.099


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the CloseFlyBy Anderson comparator summary table.")
    parser.add_argument("--src", default=Path("data/flyby/public/flyby_public_rows.csv"), type=Path)
    parser.add_argument("--dst", default=Path("data/flyby/certificates/summary.csv"), type=Path)
    args = parser.parse_args()
    path = write_anderson_summary(args.src, args.dst)
    print(path)
    return 0


__all__ = ["K_MM_PER_KMS", "anderson_mm_s", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
