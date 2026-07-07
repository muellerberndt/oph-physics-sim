from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_REPORT_NAMES = (
    "finite_repair_transition_matrix_report.json",
    "finite_certificate_report.json",
    "B_A_kernel_report.json",
    "B_A_kernel_refinement_report.json",
    "b_a_parent_report.json",
    "finite_covariant_collar_packet_parent_report.json",
    "finite_collar_boltzmann_bundle_report.json",
    "finite_collar_cmb_projection_report.json",
    "physical_scale_bridge_report.json",
    "cmb_source_provenance_report.json",
)

COMPARISON_OBSERVABLES = (
    "TT",
    "TE",
    "EE",
    "lensing",
    "BAO",
    "growth",
    "weak_lensing",
    "RSD",
    "S8",
)


@dataclass(frozen=True)
class FrozenTransferConfig:
    solver: str = "CAMB"
    solver_version_pin: str | None = None
    class_version_pin: str | None = None
    recombination_assumption: str = "Recfast/CAMB-default unless explicitly pinned by run report"
    neutrino_assumption: str = "sum_mnu_0.06eV_one_massive_two_massless"
    tolerance: float = 1.0e-5
    source_plugin_hash: str | None = None
    blinded_comparison_id: str | None = None


def frozen_transfer_likelihood_report(
    run_dirs: list[Path],
    *,
    config: FrozenTransferConfig | None = None,
) -> dict[str, Any]:
    """Validate the frozen Boltzmann-transfer and likelihood closure lane.

    This report is the simulator-side CMB1 closure object. It does not run a
    Planck likelihood itself. It verifies that the run directory contains the
    immutable source manifest, solver/recombination/neutrino pins, CDM-limit
    and Standard-Model-off regressions, blinded setup, and official likelihood
    execution receipt needed before the physical CMB gate may open.
    """

    cfg = config or FrozenTransferConfig()
    roots = _find_roots(run_dirs)
    source_manifest = _source_manifest(roots)
    cdm_regression = _first_json(roots, "cmb1_cdm_limit_regression_report.json")
    sm_off_regression = _first_json(roots, "standard_model_off_regression_report.json")
    official_execution = _first_json(roots, "official_likelihood_execution_report.json")
    official_readiness = _first_json(roots, "official_planck_likelihood_readiness_report.json")
    camb_baseline = _first_json(roots, "camb_lcdm_baseline_report.json")
    finite_collar = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    physical_input_validation = _first_json(roots, "physical_cmb_input_validation.json")
    solver = _solver_contract(cfg, camb_baseline, official_readiness, official_execution)
    likelihood = _likelihood_contract(official_execution, official_readiness)
    cdm = _cdm_regression_contract(cdm_regression, camb_baseline, tolerance=float(cfg.tolerance))
    sm_off = _standard_model_off_contract(sm_off_regression, tolerance=float(cfg.tolerance))
    source_freeze_receipt = bool(
        source_manifest["source_report_count"] > 0
        and source_manifest["all_hashes_present"]
        and source_manifest["complete_source_allowlist"]
    )
    solver_pin_receipt = bool(solver["SOLVER_ASSUMPTION_PIN_RECEIPT"])
    likelihood_hash_valid = "likelihood_hash_missing" not in set(likelihood["blockers"])
    likelihood_protocol_receipt = bool(
        source_freeze_receipt
        and solver_pin_receipt
        and _valid_sha256_hash(source_manifest["manifest_hash"])
        and _valid_sha256_hash(solver["solver_hash"])
        and _valid_sha256_hash(likelihood["likelihood_hash"])
        and likelihood_hash_valid
    )
    blinded_receipt = bool(likelihood["BLINDED_COMPARISON_SETUP_RECEIPT"])
    full_observable_receipt = bool(likelihood["FULL_OBSERVABLE_LIKELIHOOD_RECEIPT"])
    physical_boltzmann_export_receipt = bool(finite_collar.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False))
    physical_input_contract_receipt = bool(
        physical_input_validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
    )
    blockers = _unique_strings(
        [
            *([] if source_freeze_receipt else ["source_freeze_manifest_not_certified"]),
            *([] if solver_pin_receipt else ["solver_assumption_pin_not_certified"]),
            *([] if physical_boltzmann_export_receipt else ["physical_boltzmann_export_not_certified"]),
            *([] if physical_input_contract_receipt else ["physical_cmb_input_contract_not_certified"]),
            *([] if cdm["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"] else ["custom_parent_cdm_limit_regression_not_passed"]),
            *([] if sm_off["STANDARD_MODEL_OFF_REGRESSION_RECEIPT"] else ["standard_model_off_regression_not_passed"]),
            *([] if blinded_receipt else ["blinded_comparison_setup_not_certified"]),
            *([] if full_observable_receipt else ["full_observable_likelihood_not_executed"]),
            *([] if likelihood_protocol_receipt else ["frozen_likelihood_protocol_not_certified"]),
            *solver["blockers"],
            *cdm["blockers"],
            *sm_off["blockers"],
            *likelihood["blockers"],
        ]
    )
    closure_receipt = bool(
        not blockers
        and source_freeze_receipt
        and solver_pin_receipt
        and physical_boltzmann_export_receipt
        and physical_input_contract_receipt
        and cdm["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"]
        and sm_off["STANDARD_MODEL_OFF_REGRESSION_RECEIPT"]
        and blinded_receipt
        and full_observable_receipt
        and likelihood_protocol_receipt
    )
    frozen_physical_spectrum_receipt = bool(
        source_freeze_receipt
        and solver_pin_receipt
        and cdm["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"]
        and sm_off["STANDARD_MODEL_OFF_REGRESSION_RECEIPT"]
        and physical_boltzmann_export_receipt
        and physical_input_contract_receipt
    )
    likelihood_evaluated_prediction_receipt = bool(
        frozen_physical_spectrum_receipt
        and blinded_receipt
        and full_observable_receipt
        and likelihood_protocol_receipt
    )
    return {
        "mode": "frozen_transfer_likelihood_closure_v0",
        "run_dirs": [str(path) for path in roots],
        "config": asdict(cfg),
        "source_manifest": source_manifest,
        "solver_contract": solver,
        "cdm_limit_regression": cdm,
        "standard_model_off_regression": sm_off,
        "official_likelihood_execution": likelihood,
        "upstream_gate_summary": {
            "finite_collar_physical_certificate": physical_boltzmann_export_receipt,
            "physical_cmb_input_contract_receipt": physical_input_contract_receipt,
        },
        "FROZEN_SOURCE_MANIFEST_RECEIPT": source_freeze_receipt,
        "SOLVER_ASSUMPTION_PIN_RECEIPT": solver_pin_receipt,
        "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT": bool(
            cdm["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"]
        ),
        "CDM_LIMIT_REGRESSION_RECEIPT": bool(cdm["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"]),
        "STANDARD_MODEL_OFF_REGRESSION_RECEIPT": bool(sm_off["STANDARD_MODEL_OFF_REGRESSION_RECEIPT"]),
        "BLINDED_COMPARISON_SETUP_RECEIPT": blinded_receipt,
        "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT": full_observable_receipt,
        "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT": likelihood_protocol_receipt,
        "FROZEN_PHYSICAL_SPECTRUM_RECEIPT": frozen_physical_spectrum_receipt,
        "LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT": likelihood_evaluated_prediction_receipt,
        "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT": closure_receipt,
        "prediction_class": (
            "LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION"
            if likelihood_evaluated_prediction_receipt
            else (
                "FROZEN_PHYSICAL_SPECTRUM"
                if frozen_physical_spectrum_receipt
                else "DIAGNOSTIC_OR_CONDITIONAL"
            )
        ),
        "frozen_source_hash": source_manifest["manifest_hash"],
        "frozen_solver_hash": solver["solver_hash"],
        "frozen_likelihood_hash": likelihood["likelihood_hash"],
        "source_hash": source_manifest["manifest_hash"],
        "solver_hash": solver["solver_hash"],
        "likelihood_hash": likelihood["likelihood_hash"],
        "official_likelihood_execution_ready": bool(likelihood["OFFICIAL_LIKELIHOOD_EXECUTION_RECEIPT"]),
        "physical_cmb_prediction": False,
        "blockers": blockers,
        "claim_boundary": (
            "Frozen CMB1 Boltzmann-transfer and likelihood closure lane. It freezes source-side OPH "
            "inputs, pins solver/recombination/neutrino assumptions, requires CDM-limit and "
            "Standard-Model-off regressions, and admits official TT/TE/EE/lensing/BAO/growth/"
            "weak-lensing/RSD/S8 likelihood comparison only after a blinded setup receipt. Passing this "
            "report makes the hard input contract eligible; it is not itself a published CMB prediction."
        ),
    }


def write_frozen_transfer_likelihood_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    config: FrozenTransferConfig | None = None,
) -> dict[str, Any]:
    report = frozen_transfer_likelihood_report(run_dirs, config=config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_transfer_likelihood_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "frozen_transfer_likelihood_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "frozen_source_manifest.csv", report["source_manifest"]["reports"])
    return report


def _source_manifest(roots: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        for name in SOURCE_REPORT_NAMES:
            for path in _candidate_paths(root, name):
                key = path.resolve()
                if key in seen or not path.exists() or not path.is_file():
                    continue
                seen.add(key)
                rows.append(
                    {
                        "name": name,
                        "path": str(path),
                        "sha256": _sha256_file(path),
                        "source_side": True,
                    }
                )
    payload = {row["name"]: row["sha256"] for row in rows}
    manifest_hash = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "reports": rows,
        "source_report_count": len(rows),
        "required_source_names": list(SOURCE_REPORT_NAMES),
        "present_source_names": sorted({row["name"] for row in rows}),
        "missing_source_names": [name for name in SOURCE_REPORT_NAMES if name not in {row["name"] for row in rows}],
        "all_hashes_present": all(_valid_sha256_digest(row["sha256"]) for row in rows),
        "complete_source_allowlist": all(name in {row["name"] for row in rows} for name in SOURCE_REPORT_NAMES),
        "manifest_hash": manifest_hash,
        "claim_boundary": "Source-side allowlist only. CAMB, likelihood, and measurement reports are excluded.",
    }


def _solver_contract(
    config: FrozenTransferConfig,
    camb_baseline: dict[str, Any],
    official_readiness: dict[str, Any],
    official_execution: dict[str, Any],
) -> dict[str, Any]:
    solver = str(config.solver or "CAMB").upper()
    software = {}
    for report in (official_execution, camb_baseline, official_readiness):
        candidate = report.get("software") if isinstance(report.get("software"), dict) else {}
        if candidate:
            software.update(candidate)
    observed_camb = str(software.get("camb_version") or "unknown")
    observed_class = str(software.get("class_version") or "unknown")
    observed_version = observed_camb if solver == "CAMB" else observed_class
    pin = config.solver_version_pin if solver == "CAMB" else config.class_version_pin
    source_plugin_hash = str(
        config.source_plugin_hash
        or official_execution.get("source_plugin_hash")
        or camb_baseline.get("source_plugin_hash")
        or ""
    )
    blockers: list[str] = []
    if solver not in {"CAMB", "CLASS"}:
        blockers.append("unsupported_solver")
    if not pin:
        blockers.append("solver_version_pin_missing")
    elif observed_version not in {"unknown", "not_installed"} and str(pin) != observed_version:
        blockers.append("solver_version_pin_mismatch")
    if solver == "CAMB" and observed_camb in {"unknown", "not_installed"}:
        blockers.append("camb_version_not_observed")
    if solver == "CLASS" and observed_class in {"unknown", "not_installed"}:
        blockers.append("class_version_not_observed")
    if not config.recombination_assumption.strip():
        blockers.append("recombination_assumption_missing")
    if not config.neutrino_assumption.strip():
        blockers.append("neutrino_assumption_missing")
    if not _valid_sha256_hash(source_plugin_hash):
        blockers.append("source_plugin_hash_missing")
    if not np.isfinite(float(config.tolerance)) or float(config.tolerance) <= 0.0:
        blockers.append("solver_tolerance_invalid")
    payload = {
        "solver": solver,
        "solver_version_pin": pin,
        "observed_version": observed_version,
        "recombination_assumption": config.recombination_assumption,
        "neutrino_assumption": config.neutrino_assumption,
        "tolerance": float(config.tolerance),
        "source_plugin_hash": source_plugin_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    return {
        **payload,
        "software": software,
        "solver_hash": "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "SOLVER_ASSUMPTION_PIN_RECEIPT": not blockers,
        "blockers": blockers,
    }


def _cdm_regression_contract(
    cdm_regression: dict[str, Any],
    camb_baseline: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    direct = bool(cdm_regression.get("CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT", False))
    solver_native = bool(cdm_regression.get("solver_native_cdm_receipt", False))
    custom_parent = bool(cdm_regression.get("custom_parent_cdm_receipt", False))
    baseline_ok = bool(camb_baseline.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False))
    max_delta = _float(cdm_regression.get("max_relative_tt_delta"))
    delta_ok = bool(max_delta is not None and max_delta <= float(tolerance))
    receipt = bool(direct and solver_native and custom_parent and baseline_ok and delta_ok)
    blockers = []
    if not direct:
        blockers.append("cmb1_cdm_limit_regression_receipt_missing")
    if not solver_native:
        blockers.append("solver_native_cdm_regression_missing")
    if not custom_parent:
        blockers.append("custom_parent_cdm_regression_missing")
    if not baseline_ok:
        blockers.append("camb_lcdm_baseline_receipt_missing")
    if not delta_ok:
        blockers.append("cdm_limit_regression_tolerance_failed")
    return {
        "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT": receipt,
        "solver_native_cdm_receipt": solver_native,
        "custom_parent_cdm_receipt": custom_parent,
        "camb_lcdm_baseline_receipt": baseline_ok,
        "max_relative_tt_delta": max_delta,
        "tolerance": float(tolerance),
        "blockers": blockers,
    }


def _standard_model_off_contract(report: dict[str, Any], *, tolerance: float) -> dict[str, Any]:
    direct = bool(report.get("STANDARD_MODEL_OFF_REGRESSION_RECEIPT", False))
    sm_off = bool(report.get("standard_model_sector_off", False))
    anomaly_off = bool(report.get("oph_anomaly_sector_off", True))
    max_delta = _float(report.get("max_relative_tt_delta", report.get("max_regression_delta")))
    delta_ok = bool(max_delta is not None and max_delta <= float(tolerance))
    no_particle_sources = bool(report.get("no_particle_sector_sources", False))
    receipt = bool(direct and sm_off and anomaly_off and no_particle_sources and delta_ok)
    blockers = []
    if not direct:
        blockers.append("standard_model_off_regression_receipt_missing")
    if not sm_off:
        blockers.append("standard_model_sector_not_off")
    if not anomaly_off:
        blockers.append("anomaly_sector_not_off_for_control")
    if not no_particle_sources:
        blockers.append("particle_sector_sources_present_in_sm_off_control")
    if not delta_ok:
        blockers.append("standard_model_off_regression_tolerance_failed")
    return {
        "STANDARD_MODEL_OFF_REGRESSION_RECEIPT": receipt,
        "standard_model_sector_off": sm_off,
        "oph_anomaly_sector_off": anomaly_off,
        "no_particle_sector_sources": no_particle_sources,
        "max_relative_tt_delta": max_delta,
        "tolerance": float(tolerance),
        "blockers": blockers,
    }


def _likelihood_contract(official_execution: dict[str, Any], official_readiness: dict[str, Any]) -> dict[str, Any]:
    readiness = bool(
        official_readiness.get("official_likelihood_execution_ready", False)
        or official_execution.get("official_likelihood_execution_ready", False)
    )
    execution_receipt = bool(official_execution.get("OFFICIAL_LIKELIHOOD_EXECUTION_RECEIPT", False))
    blinded = bool(
        official_execution.get("BLINDED_COMPARISON_SETUP_RECEIPT", False)
        or official_execution.get("blinded_comparison_setup_receipt", False)
    )
    observable_rows = official_execution.get("observables")
    if not isinstance(observable_rows, dict):
        observable_rows = {}
    observable_receipts = {
        name: bool(observable_rows.get(name, False) or official_execution.get(f"{name}_LIKELIHOOD_RECEIPT", False))
        for name in COMPARISON_OBSERVABLES
    }
    full = bool(
        official_execution.get("FULL_OBSERVABLE_LIKELIHOOD_RECEIPT", False)
        or all(observable_receipts.values())
    )
    likelihood_hash = str(
        official_execution.get("likelihood_hash")
        or official_execution.get("frozen_likelihood_hash")
        or official_readiness.get("likelihood_hash")
        or ""
    )
    blockers = []
    if not readiness:
        blockers.append("official_likelihood_environment_not_ready")
    if not execution_receipt:
        blockers.append("official_likelihood_execution_receipt_missing")
    if not blinded:
        blockers.append("blinded_comparison_setup_receipt_missing")
    if not full:
        blockers.append("full_observable_likelihood_receipt_missing")
    if not _valid_sha256_hash(likelihood_hash):
        blockers.append("likelihood_hash_missing")
    return {
        "official_likelihood_execution_ready": readiness,
        "OFFICIAL_LIKELIHOOD_EXECUTION_RECEIPT": execution_receipt,
        "BLINDED_COMPARISON_SETUP_RECEIPT": blinded,
        "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT": full,
        "observable_receipts": observable_receipts,
        "missing_observables": [name for name, passed in observable_receipts.items() if not passed],
        "likelihood_hash": likelihood_hash if _valid_sha256_hash(likelihood_hash) else None,
        "blockers": blockers,
    }


def _find_roots(run_dirs: list[Path]) -> list[Path]:
    return [Path(path) for path in run_dirs if Path(path).exists()]


def _candidate_paths(root: Path, name: str) -> list[Path]:
    if root.is_file() and root.name == name:
        return [root]
    if root.is_dir():
        return [root / name, *sorted(root.glob(f"**/{name}"))]
    return []


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    for root in roots:
        for path in _candidate_paths(Path(root), name):
            if not path.exists() or not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data:
                return data
    return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256_digest(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _valid_sha256_hash(value: str | None) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    return _valid_sha256_digest(value.removeprefix("sha256:"))


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen Transfer/Likelihood Closure",
        "",
        str(report.get("claim_boundary", "")),
        "",
        "## Receipts",
        "",
        f"- source freeze manifest: `{str(report.get('FROZEN_SOURCE_MANIFEST_RECEIPT', False)).lower()}`",
        f"- solver assumption pins: `{str(report.get('SOLVER_ASSUMPTION_PIN_RECEIPT', False)).lower()}`",
        f"- CDM-limit regression: `{str(report.get('CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT', False)).lower()}`",
        f"- Standard-Model-off regression: `{str(report.get('STANDARD_MODEL_OFF_REGRESSION_RECEIPT', False)).lower()}`",
        f"- blinded comparison setup: `{str(report.get('BLINDED_COMPARISON_SETUP_RECEIPT', False)).lower()}`",
        f"- full observable likelihood: `{str(report.get('FULL_OBSERVABLE_LIKELIHOOD_RECEIPT', False)).lower()}`",
        f"- closure: `{str(report.get('FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT', False)).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
