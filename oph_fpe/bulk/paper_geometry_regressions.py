"""Golden paper-side geometry fixtures, kept separate from run receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.evidence.cross_repo_artifacts import verify_cross_repo_artifact_manifest


GEOMETRY_KEYS = (
    "geometry_cyclic_cap_net",
    "geometry_modular_clock",
    "geometry_null_net",
    "geometry_realized_events",
    "geometry_bulk_depth",
)
DEFAULT_GEOMETRY_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "data" / "oph_cross_repo_current"


def paper_geometry_regression_report(import_root: Path | None = None) -> dict[str, Any]:
    root = Path(import_root) if import_root is not None else DEFAULT_GEOMETRY_FIXTURE_ROOT
    verification = verify_cross_repo_artifact_manifest(root)
    rows = {
        str(row.get("key")): row
        for row in verification.get("manifest", {}).get("artifacts", [])
        if isinstance(row, dict) and row.get("key") in GEOMETRY_KEYS and row.get("hash_verified") is True
    }
    payloads = {key: _read(root / str(row["target_relpath"])) for key, row in rows.items()}
    cyclic = payloads.get("geometry_cyclic_cap_net", {})
    modular = payloads.get("geometry_modular_clock", {})
    null_net = payloads.get("geometry_null_net", {})
    events = payloads.get("geometry_realized_events", {})
    bulk = payloads.get("geometry_bulk_depth", {})

    fixture_receipts = {
        "cyclic_cap_net": _all_literal_true(cyclic.get("receipts_witnessed")),
        "modular_clock_boundary_collar": _all_literal_true(modular.get("receipts_witnessed")),
        "null_net_one_particle": _all_literal_true(null_net.get("receipts_witnessed")),
        "event_sheet_countermodel": _event_countermodel(events),
        "bulk_depth_seed_sweep": _all_literal_true(bulk.get("receipts_witnessed")),
        "bulk_depth_2_2_negative_control": _nested(bulk, "countermodel_global_coupling", "signature")
        == [2, 2],
    }
    open_obligations = {
        key: [str(value) for value in payload.get("receipts_pending", [])]
        for key, payload in payloads.items()
        if isinstance(payload.get("receipts_pending"), list)
    }
    caveats = [str(value) for value in bulk.get("caveats", [])] if isinstance(bulk.get("caveats"), list) else []
    return {
        "schema": "oph_paper_geometry_regression_report_v1",
        "artifact_manifest_verified": verification.get("verified") is True,
        "fixtures_present": sorted(rows),
        "all_fixtures_present": set(rows) == set(GEOMETRY_KEYS),
        "fixture_receipts": fixture_receipts,
        "all_golden_regressions_pass": bool(
            set(rows) == set(GEOMETRY_KEYS) and all(fixture_receipts.values())
        ),
        "open_obligations": open_obligations,
        "bulk_depth_caveats": caveats,
        "timelike_eigenvalues_across_seeds": list(bulk.get("timelike_eigenvalues_across_seeds") or []),
        "paper_side_fixtures_informational": True,
        "sim_native_geometry_receipt": False,
        "einstein_branch_entry_receipt": False,
        "claim_boundary": (
            "Passing these hash-pinned golden fixtures checks simulator/paper interface drift. "
            "It does not show that the current simulator run produced the paper-side states or entered the Einstein branch."
        ),
    }


def write_paper_geometry_regression_report(
    import_root: Path | None = None, out: Path | None = None
) -> dict[str, Any]:
    report = paper_geometry_regression_report(import_root)
    root = Path(import_root) if import_root is not None else DEFAULT_GEOMETRY_FIXTURE_ROOT
    target = Path(out) if out is not None else root / "paper_geometry_regression_report.json"
    if target.is_dir():
        target = target / "paper_geometry_regression_report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _event_countermodel(payload: dict[str, Any]) -> bool:
    receipts = payload.get("receipts_witnessed")
    if not isinstance(receipts, dict):
        return False
    positive = [value for key, value in receipts.items() if key != "e3_rank_four_bulk_depth"]
    return bool(positive and all(value is True for value in positive) and receipts.get("e3_rank_four_bulk_depth") is False)


def _all_literal_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
