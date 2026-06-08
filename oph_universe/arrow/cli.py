from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from oph_universe.arrow.scenarios import run_scenario
from oph_universe.arrow.schemas import write_json, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oph_universe.arrow.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--seed", default=1, type=int)
    run.add_argument("--out", required=True, type=Path)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--run-dir", required=True, type=Path)
    plot = sub.add_parser("plot")
    plot.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        config = _read_yaml(args.config)
        result = run_scenario(config, seed=args.seed)
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
        write_jsonl(out / "metrics.jsonl", result.metrics)
        write_jsonl(out / "checkpoints.jsonl", result.checkpoints)
        write_jsonl(out / "records.jsonl", result.records)
        write_json(out / "ancestry_candidates.json", result.candidates)
        write_json(out / "selected_ancestry.json", result.selected)
        write_json(out / "summary.json", result.summary)
        print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "summarize":
        summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "plot":
        # Plotting is intentionally optional; emit a stable placeholder summary
        # when matplotlib is not installed.
        write_json(args.run_dir / "plot_manifest.json", {"plotting_available": False, "reason": "not implemented in MVP"})
        print(json.dumps({"plot_manifest": str(args.run_dir / "plot_manifest.json")}, indent=2))
        return 0
    return 2


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    raise SystemExit(main())
