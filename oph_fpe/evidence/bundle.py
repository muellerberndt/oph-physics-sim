from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from oph_fpe.dynamics.repair import RepairEvent
from oph_fpe.evidence.hashes import stable_json_hash


class RunBundle:
    def __init__(self, root: Path, run_id: str):
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "controls").mkdir()
        (self.path / "plots").mkdir()

    def write_config(self, config: dict[str, Any]) -> None:
        (self.path / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (self.path / "seed_material.json").write_text(
            json.dumps({"config_hash": stable_json_hash(config), "seed": config.get("seed")}, indent=2),
            encoding="utf-8",
        )

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["git_commit"] = _git_commit()
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        (self.path / "git_commit.txt").write_text(str(manifest["git_commit"]) + "\n", encoding="utf-8")

    def write_json(self, name: str, data: Any) -> None:
        (self.path / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.path / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")

    def write_mismatch_trace(self, rows: list[dict[str, Any]]) -> None:
        _write_csv(self.path / "mismatch_trace.csv", rows)

    def write_repair_events(self, events: list[RepairEvent]) -> None:
        _write_csv(
            self.path / "repair_events.csv",
            [
                {
                    "cycle": event.cycle,
                    "node": event.node,
                    "beta": event.beta,
                    "phi_before": event.phi_before,
                    "phi_after": event.phi_after,
                    "delta_phi": event.delta_phi,
                    "accepted": event.accepted,
                    "reason": event.reason,
                }
                for event in events
            ],
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
