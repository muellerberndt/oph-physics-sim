from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from scripts.ga71_td._pending import pending_main


if __name__ == "__main__":
    raise SystemExit(pending_main("freeze_inputs"))
