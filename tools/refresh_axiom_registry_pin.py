#!/usr/bin/env python3
"""Refresh the simulator's verbatim three-axiom pin from a clean RER checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SIM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THEORY_ROOT = SIM_ROOT.parent / "reverse-engineering-reality"
DEFAULT_OUTPUT = SIM_ROOT / "data/theory/axiom_registry_pin.json"
PIN_FIELDS = ("id", "key", "title", "informal", "formal_concise", "reference_anchor")
CLAIM_BOUNDARY = (
    "Verbatim copies of the canonical axiom statements from the theory "
    "repository's machine registry, pinned by commit and content hash. The "
    "simulator realizes finite architectural fragments of these axioms; "
    "carrying the statements promotes no physical claim."
)


def _git(theory_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(theory_root), *args], text=True
    ).strip()


def _load_registry(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "{")
    except StopIteration as exc:
        raise ValueError(f"canonical registry has no JSON object: {path}") from exc
    registry = json.loads("\n".join(lines[start:]))
    ids = [axiom.get("id") for axiom in registry.get("axioms", [])]
    if ids != ["A1", "A2", "A3"] or registry.get("core_axiom_count") != 3:
        raise ValueError(f"canonical registry must carry exactly A1, A2, A3; found {ids}")
    return registry


def _release_id(theory_root: Path) -> str:
    release_info = theory_root / "paper/release_info.tex"
    match = re.search(
        r"\\newcommand\{\\OPHPaperReleaseID\}\{([^}]+)\}",
        release_info.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"cannot read OPH release id from {release_info}")
    return match.group(1)


def build_pin(theory_root: Path) -> dict[str, Any]:
    registry_path = theory_root / "claims/axiom_registry.yaml"
    relative_path = "claims/axiom_registry.yaml"
    if subprocess.run(
        ["git", "-C", str(theory_root), "diff", "--quiet", "--", relative_path],
        check=False,
    ).returncode != 0:
        raise ValueError("canonical axiom registry has uncommitted changes")
    registry = _load_registry(registry_path)
    return {
        "schema": "oph.sim.axiom_registry_pin.v1",
        "source": {
            "repository": "reverse-engineering-reality",
            "path": relative_path,
            "commit": _git(theory_root, "rev-parse", "HEAD"),
            "release": _release_id(theory_root),
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
        "core_axiom_count": 3,
        "axioms": [
            {field: axiom[field] for field in PIN_FIELDS}
            for axiom in registry["axioms"]
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-root", type=Path, default=DEFAULT_THEORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pin = build_pin(args.theory_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(pin, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"pinned {pin['source']['commit']} {pin['source']['release']} "
        f"sha256:{pin['source']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
