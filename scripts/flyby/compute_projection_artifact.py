from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from scripts.flyby._pending import pending_main


if __name__ == "__main__":
    raise SystemExit(pending_main("compute_projection_artifact"))
