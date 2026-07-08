from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # PyYAML is a project dependency, but keep text fallback for code-only bundles.
    import yaml
except Exception:  # pragma: no cover
    yaml = None


CLAIM_TIERS = (
    "J0_DIAGNOSTIC_PROXY",
    "J1_CATALOG_RECORD",
    "J2_SPECTROSCOPIC_OR_PHOTOMETRIC_OBJECT",
    "J3_DEGENERACY_AUDITED_OBJECT",
    "J4_CONDITIONAL_PHYSICAL_OBJECT",
    "J5_SOURCE_RELEASE_CANDIDATE",
    "J6_SOURCE_ONLY_OBJECT_ABUNDANCE",
    "J7_FORWARD_MOCK_PHYSICAL_SPECTRUM",
    "J8_LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION",
)

DEGENERACY_RECEIPTS = (
    "REDSHIFT_RECEIPT",
    "DUST_RECEIPT",
    "AGN_RECEIPT",
    "NEBULAR_RECEIPT",
    "STELLAR_POPULATION_RECEIPT",
    "LENSING_RECEIPT",
    "MORPHOLOGY_PSF_RECEIPT",
    "SELECTION_RECEIPT",
)

CLAIM_REQUIREMENTS = {
    "J1_CATALOG_RECORD": ("CATALOG_INGESTION_RECEIPT", "CATALOG_PROVENANCE_RECEIPT"),
    "J2_SPECTROSCOPIC_OR_PHOTOMETRIC_OBJECT": (
        "OBJECT_ID_RECEIPT",
        "APERTURE_RECEIPT",
        "REDSHIFT_POSTERIOR_RECEIPT",
        "PHOTOMETRY_OR_SPECTRUM_RECEIPT",
        "MORPHOLOGY_RECORD_RECEIPT",
    ),
    "J3_DEGENERACY_AUDITED_OBJECT": DEGENERACY_RECEIPTS + ("DEGENERACY_AUDIT_RECEIPT",),
    "J4_CONDITIONAL_PHYSICAL_OBJECT": (
        "OBJECT_PARENT_RECEIPT",
        "PACKET_MASS_SHELL_RECEIPT",
        "FINITE_PACKET_STRESS_READOUT_RECEIPT",
        "TOTAL_STRESS_CLOSURE_RECEIPT",
        "RADIATIVE_TRANSFER_RECEIPT",
        "LENSING_SOURCE_PLANE_RECEIPT",
        "CHEMICAL_SFH_RECEIPT",
    ),
    "J5_SOURCE_RELEASE_CANDIDATE": (
        "OBJECT_RELEASE_STATE_RECEIPT",
        "SOURCE_RELEASE_RESIDUAL_RECEIPT",
    ),
    "J6_SOURCE_ONLY_OBJECT_ABUNDANCE": (
        "OBJECT_QUOTIENT_ENSEMBLE_RECEIPT",
        "OBJECT_SOURCE_LAW_RECEIPT",
        "OBJECT_ABUNDANCE_SOURCE_RECEIPT",
        "NO_TARGET_LEAKAGE_RECEIPT",
        "LOAD_REFINEMENT_COMPATIBILITY_RECEIPT",
    ),
    "J7_FORWARD_MOCK_PHYSICAL_SPECTRUM": (
        "JWST_FORWARD_OPERATOR_RECEIPT",
        "JWST_SELECTION_RECEIPT",
        "FROZEN_FORWARD_OPERATOR_RECEIPT",
    ),
    "J8_LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION": (
        "FROZEN_SOURCE_RECEIPT",
        "FROZEN_CATALOG_DATA_RECEIPT",
        "FROZEN_CATALOG_LIKELIHOOD_RECEIPT",
        "JWST_LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT",
    ),
}

NONCLAIMS = (
    "compact record surface is not physical assembly",
    "red color is not old age",
    "luminosity is not stellar mass",
    "broad lines are not automatically large black-hole mass",
    "source release is not galaxy formation",
    "record density is not stellar-mass density",
)

TARGET_DATA_TOKENS = (
    "jwst_catalog",
    "catalog_count",
    "catalog_counts",
    "observed_count",
    "observed_counts",
    "anomaly_label",
    "interesting_object",
    "posterior_summary",
    "likelihood_residual",
    "model_comparison_score",
    "mass_age_tension_label",
)

REQUIRED_SCHEMA_FILES = (
    "schemas/jwst/object_quotient_ensemble.schema.json",
    "schemas/jwst/object_source_law.schema.json",
    "schemas/jwst/object_release_state.schema.json",
    "schemas/jwst/compact_record_surface.schema.json",
    "schemas/jwst/object_parent.schema.json",
    "schemas/jwst/jwst_forward_operator.schema.json",
    "schemas/jwst/catalog_ledger_row.schema.json",
    "schemas/jwst/degeneracy_audit.schema.json",
    "schemas/jwst/object_abundance_receipt.schema.json",
    "schemas/jwst/frozen_catalog_likelihood.schema.json",
)


def write_source_artifact_report(out_dir: Path, *, config: Path | None = None) -> dict[str, Any]:
    leak_hits = _target_leak_hits(config)
    receipts = {
        "OBJECT_QUOTIENT_ENSEMBLE_RECEIPT": False,
        "OBJECT_SOURCE_LAW_RECEIPT": False,
        "OBJECT_RELEASE_STATE_RECEIPT": False,
        "NO_TARGET_LEAKAGE_RECEIPT": not leak_hits,
    }
    report = _stage_report(
        "jwst_object_source_artifact_v1",
        "object_source_artifact",
        receipts,
        inputs={"config": _path_meta(config), "target_leak_hits": leak_hits},
        next_command="jwst-source-sample",
    )
    return _write_report_pair(out_dir, "jwst_object_source_artifact_report", report)


def write_source_sample_report(out_dir: Path, *, source: Path, n_samples: int) -> dict[str, Any]:
    source_report = _read_report(source)
    source_receipts = _extract_receipts(source_report)
    source_ready = bool(
        source_receipts.get("OBJECT_QUOTIENT_ENSEMBLE_RECEIPT")
        and source_receipts.get("OBJECT_SOURCE_LAW_RECEIPT")
        and source_receipts.get("NO_TARGET_LEAKAGE_RECEIPT")
    )
    receipts = {
        "SAMPLER_CORRECTNESS_RECEIPT": False,
        "QUOTIENT_LUMPABILITY_RECEIPT": False,
        "REPRESENTATIVE_LIFT_RECEIPT": False,
        "REFINEMENT_COMPATIBILITY_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_source_sample_v1",
        "source_sample",
        receipts,
        inputs={"source": _path_meta(source), "n_samples": int(n_samples), "source_ready": source_ready},
        blockers=[] if source_ready else ["source_artifact_not_promoted"],
        next_command="jwst-compact-record-surfaces",
    )
    return _write_report_pair(out_dir, "jwst_source_sample_report", report)


def write_compact_record_surface_report(out_dir: Path, *, samples: Path) -> dict[str, Any]:
    receipts = {
        "COMPACT_RECORD_SURFACE_RECEIPT": False,
        "SYNC_RESIDUAL_RECEIPT": False,
        "RECORD_DENSITY_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_compact_record_surface_v1",
        "compact_record_surface",
        receipts,
        inputs={"samples": _path_meta(samples)},
        next_command="jwst-object-parent",
    )
    return _write_report_pair(out_dir, "jwst_compact_record_surface_report", report)


def write_object_parent_report(
    out_dir: Path,
    *,
    samples: Path,
    parent_config: Path | None = None,
) -> dict[str, Any]:
    receipts = {
        "OBJECT_PARENT_RECEIPT": False,
        "PACKET_MASS_SHELL_RECEIPT": False,
        "FINITE_PACKET_STRESS_READOUT_RECEIPT": False,
        "TOTAL_STRESS_CLOSURE_RECEIPT": False,
        "RADIATIVE_TRANSFER_RECEIPT": False,
        "LENSING_SOURCE_PLANE_RECEIPT": False,
        "CHEMICAL_SFH_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_object_parent_v1",
        "object_parent",
        receipts,
        inputs={"samples": _path_meta(samples), "parent_config": _path_meta(parent_config)},
        next_command="jwst-forward-mock",
    )
    return _write_report_pair(out_dir, "jwst_object_parent_report", report)


def write_forward_mock_report(
    out_dir: Path,
    *,
    parent: Path,
    instrument: Path | None = None,
) -> dict[str, Any]:
    receipts = {
        "JWST_FORWARD_OPERATOR_RECEIPT": False,
        "JWST_SELECTION_RECEIPT": False,
        "CATALOG_EXTRACTION_RECEIPT": False,
        "FROZEN_FORWARD_OPERATOR_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_forward_mock_v1",
        "forward_mock",
        receipts,
        inputs={"parent": _path_meta(parent), "instrument": _path_meta(instrument)},
        next_command="jwst-degeneracy-audit",
    )
    return _write_report_pair(out_dir, "jwst_forward_mock_report", report)


def write_degeneracy_audit_report(
    out_dir: Path,
    *,
    catalog: Path,
    models: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl_or_empty(catalog)
    synthetic_pairs = [row for row in rows if bool(row.get("synthetic_degeneracy_pair"))]
    overpromoted = [
        str(row.get("object_id", index))
        for index, row in enumerate(rows)
        if row.get("claim_label") == "PHYSICAL_MASS_AGE_TENSION"
        and not _row_degeneracy_closed(row)
    ]
    receipts = {name: False for name in DEGENERACY_RECEIPTS}
    receipts.update(
        {
            "DEGENERACY_AUDIT_RECEIPT": False,
            "MASS_AGE_TENSION_PROMOTION_RECEIPT": False,
            "SYNTHETIC_DEGENERACY_PAIR_GUARD_RECEIPT": not overpromoted,
        }
    )
    report = _stage_report(
        "jwst_degeneracy_audit_v1",
        "degeneracy_audit",
        receipts,
        inputs={"catalog": _path_meta(catalog), "models": _path_meta(models), "row_count": len(rows)},
        extra={
            "synthetic_degeneracy_pair_count": len(synthetic_pairs),
            "overpromoted_object_ids": overpromoted,
            "recommended_label_when_open": "DEGENERACY_OPEN",
        },
        blockers=["degeneracy_overpromotion_detected"] if overpromoted else None,
        next_command="jwst-object-abundance-selector",
    )
    return _write_report_pair(out_dir, "jwst_degeneracy_audit_report", report)


def write_abundance_selector_report(
    out_dir: Path,
    *,
    source: Path,
    samples: Path,
    bins: Path | None = None,
) -> dict[str, Any]:
    receipts = {
        "OBJECT_ABUNDANCE_SOURCE_RECEIPT": False,
        "LOAD_REFINEMENT_COMPATIBILITY_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_object_abundance_selector_v1",
        "abundance_selector",
        receipts,
        inputs={"source": _path_meta(source), "samples": _path_meta(samples), "bins": _path_meta(bins)},
        next_command="jwst-frozen-catalog-likelihood",
    )
    return _write_report_pair(out_dir, "jwst_object_abundance_selector_report", report)


def write_frozen_likelihood_report(
    out_dir: Path,
    *,
    source: Path,
    forward: Path,
    catalog_data: Path,
    likelihood: Path,
) -> dict[str, Any]:
    receipts = {
        "FROZEN_SOURCE_RECEIPT": False,
        "FROZEN_FORWARD_OPERATOR_RECEIPT": False,
        "FROZEN_CATALOG_DATA_RECEIPT": False,
        "FROZEN_CATALOG_LIKELIHOOD_RECEIPT": False,
        "JWST_LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT": False,
    }
    report = _stage_report(
        "jwst_frozen_catalog_likelihood_v1",
        "frozen_catalog_likelihood",
        receipts,
        inputs={
            "source": _path_meta(source),
            "forward": _path_meta(forward),
            "catalog_data": _path_meta(catalog_data),
            "likelihood": _path_meta(likelihood),
        },
        next_command=None,
    )
    return _write_report_pair(out_dir, "jwst_frozen_catalog_likelihood_report", report)


def write_simulation_plan_report(out_dir: Path, *, run_dir: Path) -> dict[str, Any]:
    receipts = collect_run_receipts(run_dir)
    claim, first_blocked, missing = strongest_allowed_claim(receipts)
    report = {
        "mode": "jwst_compact_object_simulation_plan_v1",
        "generated_utc": _now_utc(),
        "run_dir": str(Path(run_dir)),
        "strongest_allowed_claim": claim,
        "first_blocked_gate": first_blocked,
        "missing_receipts": missing,
        "nonclaims": list(NONCLAIMS),
        "next_required_command": _next_command(first_blocked),
        "receipt_count": len(receipts),
        "receipts": receipts,
        "claim_boundary": (
            "The planner recomputes claim tier from artifacts. Producer-supplied booleans "
            "cannot promote JWST compact-object diagnostics across semantic claim boundaries."
        ),
    }
    return _write_report_pair(out_dir, "jwst_compact_object_simulation_plan", report)


def collect_run_receipts(run_dir: Path) -> dict[str, bool]:
    receipts: dict[str, bool] = {}
    for path in Path(run_dir).rglob("*_report.json"):
        report = _read_report(path)
        for key, value in _extract_receipts(report).items():
            receipts[key] = bool(value)
    return receipts


def strongest_allowed_claim(receipts: dict[str, bool]) -> tuple[str, str | None, list[str]]:
    strongest = "J0_DIAGNOSTIC_PROXY"
    for tier in CLAIM_TIERS[1:]:
        required = CLAIM_REQUIREMENTS[tier]
        missing = [name for name in required if not bool(receipts.get(name, False))]
        if missing:
            return strongest, missing[0], missing
        strongest = tier
    return strongest, None, []


def _stage_report(
    mode: str,
    stage: str,
    receipts: dict[str, bool],
    *,
    inputs: dict[str, Any],
    blockers: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    next_command: str | None,
) -> dict[str, Any]:
    missing = [name for name, passed in receipts.items() if not bool(passed)]
    merged_blockers = list(blockers or [])
    merged_blockers.extend(f"{name}_missing" for name in missing)
    report = {
        "schema": "oph_jwst_compact_object_stage_report_v1",
        "mode": mode,
        "stage": stage,
        "generated_utc": _now_utc(),
        "receipts": receipts,
        "readiness_gates": receipts,
        "blocking_receipts": missing,
        "blockers": merged_blockers,
        "inputs": inputs,
        "nonclaims": list(NONCLAIMS),
        "next_required_command": next_command,
        "claim_boundary": (
            "Fail-closed JWST compact-object source-release workbench. This artifact is a receipt "
            "surface, not evidence that JWST has confirmed OPH."
        ),
    }
    if extra:
        report.update(extra)
    return report


def _write_report_pair(out_dir: Path, stem: str, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    receipts = report.get("receipts") or {}
    return "\n".join(
        [
            f"# {str(report.get('stage', 'jwst')).replace('_', ' ').title()}",
            "",
            str(report.get("claim_boundary", "")),
            "",
            f"- mode: `{report.get('mode')}`",
            f"- strongest allowed claim: `{report.get('strongest_allowed_claim', 'J0_DIAGNOSTIC_PROXY')}`",
            f"- next required command: `{report.get('next_required_command')}`",
            "",
            "## Receipts",
            "",
            *[f"- {key}: `{str(bool(value)).lower()}`" for key, value in receipts.items()],
            "",
            "## Blockers",
            "",
            *[f"- `{item}`" for item in report.get("blockers", report.get("missing_receipts", []))],
            "",
        ]
    )


def _row_degeneracy_closed(row: dict[str, Any]) -> bool:
    receipts = row.get("receipts") if isinstance(row.get("receipts"), dict) else row
    return all(bool(receipts.get(name, False)) for name in DEGENERACY_RECEIPTS)


def _extract_receipts(report: dict[str, Any]) -> dict[str, bool]:
    receipts: dict[str, bool] = {}
    for container_key in ("receipts", "readiness_gates"):
        container = report.get(container_key)
        if isinstance(container, dict):
            for key, value in container.items():
                if isinstance(value, bool):
                    receipts[key] = value
    for key, value in report.items():
        if key.isupper() and isinstance(value, bool):
            receipts[key] = value
    return receipts


def _read_report(path: Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        reports = sorted(source.rglob("*_report.json"))
        if not reports:
            return {}
        source = reports[0]
    if not source.is_file():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl_or_empty(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _target_leak_hits(path: Path | None) -> list[str]:
    if path is None or not Path(path).is_file():
        return []
    text = Path(path).read_text(encoding="utf-8")
    parsed: Any = None
    try:
        if Path(path).suffix.lower() == ".json":
            parsed = json.loads(text)
        elif yaml is not None and Path(path).suffix.lower() in {".yaml", ".yml"}:
            parsed = yaml.safe_load(text)
    except Exception:
        parsed = None
    haystack = json.dumps(parsed, sort_keys=True).lower() if parsed is not None else text.lower()
    return sorted(token for token in TARGET_DATA_TOKENS if token in haystack)


def _path_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    source = Path(path)
    meta: dict[str, Any] = {"path": str(source), "exists": source.exists()}
    if source.is_file():
        raw = source.read_bytes()
        meta.update({"sha256": _sha256_bytes(raw), "byte_count": len(raw)})
    return meta


def _next_command(first_blocked_gate: str | None) -> str | None:
    if first_blocked_gate is None:
        return None
    command_by_gate = {
        "CATALOG_INGESTION_RECEIPT": "jwst-degeneracy-audit",
        "OBJECT_QUOTIENT_ENSEMBLE_RECEIPT": "jwst-object-source-artifact",
        "OBJECT_SOURCE_LAW_RECEIPT": "jwst-object-source-artifact",
        "OBJECT_RELEASE_STATE_RECEIPT": "jwst-object-source-artifact",
        "OBJECT_PARENT_RECEIPT": "jwst-object-parent",
        "COMPACT_RECORD_SURFACE_RECEIPT": "jwst-compact-record-surfaces",
        "JWST_FORWARD_OPERATOR_RECEIPT": "jwst-forward-mock",
        "DEGENERACY_AUDIT_RECEIPT": "jwst-degeneracy-audit",
        "OBJECT_ABUNDANCE_SOURCE_RECEIPT": "jwst-object-abundance-selector",
        "FROZEN_CATALOG_LIKELIHOOD_RECEIPT": "jwst-frozen-catalog-likelihood",
    }
    return command_by_gate.get(first_blocked_gate, "jwst-compact-object-simulation-plan")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
