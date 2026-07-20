"""Command-line verifier for string-vacuum candidate and catalogue packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from oph_fpe.evidence.hashes import stable_json_hash

from .contract import verify_candidate_evidence, verify_catalogue_evidence
from .receipt_targets import (
    observable_target_registry,
    observable_target_registry_sha256,
    receipt_target_registry,
    receipt_target_registry_sha256,
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence packet must contain a JSON object")
    return payload


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-candidate", "verify-catalogue"):
        child = subparsers.add_parser(name)
        child.add_argument("packet", type=Path)
        child.add_argument("--bundle-root", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
    describe = subparsers.add_parser("describe-targets")
    describe.add_argument(
        "--kind",
        choices=("all", "receipts", "observables"),
        default="all",
    )
    describe.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "describe-targets":
        report: dict = {
            "artifact": "oph_string_vacuum_simulator_target_specification",
            "schema_version": 1,
        }
        if args.kind in {"all", "receipts"}:
            report["receipt_target_registry_sha256"] = receipt_target_registry_sha256()
            report["receipt_targets"] = receipt_target_registry()
        if args.kind in {"all", "observables"}:
            report["observable_target_registry_sha256"] = observable_target_registry_sha256()
            report["observable_targets"] = observable_target_registry()
        if args.out is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _write(args.out, report)
        return 0
    try:
        payload = _load(args.packet)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if args.command == "verify-candidate":
            report = verify_candidate_evidence({}, bundle_root=args.bundle_root)
        else:
            report = verify_catalogue_evidence({}, bundle_root=args.bundle_root)
        report.pop("report_sha256", None)
        blocker = f"input_packet_unreadable:{type(exc).__name__}"
        report["blockers"] = sorted(set(report.get("blockers", []) + [blocker]))
        if args.command == "verify-catalogue":
            report["catalogue_blockers"] = sorted(
                set(report.get("catalogue_blockers", []) + [blocker])
            )
        report["report_sha256"] = stable_json_hash(report)
        _write(args.out, report)
        return 1
    if args.command == "verify-candidate":
        report = verify_candidate_evidence(payload, bundle_root=args.bundle_root)
        success = report["candidate_status"] == "PASS"
    else:
        report = verify_catalogue_evidence(payload, bundle_root=args.bundle_root)
        success = report["catalogue_status"] == "SELECTED_WITHIN_DECLARED_CATALOGUE"
    _write(args.out, report)
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
