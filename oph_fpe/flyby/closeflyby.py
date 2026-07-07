from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_ROWS = REPO_ROOT / "data" / "flyby" / "public" / "flyby_public_rows.csv"
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "data" / "flyby" / "public" / "flyby_source_manifest.yaml"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "flyby" / "raw"
DEFAULT_SCHEMA = REPO_ROOT / "data" / "flyby" / "certificates" / "closeflyby_schema.json"

K_MM_PER_KMS = 3.099

STATUS_ENUM = (
    "SPEC_ONLY",
    "PUBLIC_EVIDENCE_ATTACHED",
    "OD_REPLAY_PENDING",
    "SYNTHETIC_PIPELINE_VALIDATED",
    "HISTORICAL_REPLAY_PASS",
    "HISTORICAL_REPLAY_FAILED",
    "FULL_REPLAY_PASS",
    "FULL_REPLAY_PASS_PROJECTION_UNRESOLVED",
    "SOLVED_PROJECTION_ARTIFACT",
    "SOLVED_NULL",
    "CONVENTIONAL_REPAIR_CANDIDATE",
    "DATA_INCOMPLETE_NOT_SOLVED",
    "NOT_SOLVED_RESIDUAL",
    "REJECTED_LAW",
)

RAW_RECEIPT_ARRAY_KEYS = (
    "tracking_files",
    "station_files",
    "clock_ramp_time_files",
    "eop_files",
    "tro_files",
    "ion_files",
    "weather_files",
    "spice_files",
    "attitude_files",
    "maneuver_files",
    "thermal_srp_optical_files",
    "design_matrix_files",
)

TEST_KEYS = (
    "historical_replay_pass",
    "full_replay_pass",
    "projection_pass",
    "chi2_reduced",
    "doppler_rms_mm_s",
    "range_rms_m",
    "whiteness_p_value",
    "station_holdout_pass",
    "arc_holdout_pass",
    "weighted_chi2",
    "pre_perigee_holdout_pass",
    "post_perigee_holdout_pass",
    "leave_one_pass_out_pass",
    "bias_by_station_pass",
    "bias_by_observable_type_pass",
    "covariance_condition_number",
    "parameter_correlation_report",
)


def anderson_mm_s(vinf_km_s: float, din_deg: float, dout_deg: float) -> float:
    return K_MM_PER_KMS * vinf_km_s * (
        math.cos(math.radians(din_deg)) - math.cos(math.radians(dout_deg))
    )


def load_public_rows(path: Path = DEFAULT_PUBLIC_ROWS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "dVp_obs_mm_s": _float_or_none(row["dVp_obs_mm_s"]),
                    "sigma_dVp_mm_s": _float_or_none(row["sigma_dVp_mm_s"]),
                    "dVinf_obs_mm_s": _float_or_none(row["dVinf_obs_mm_s"]),
                    "sigma_dVinf_mm_s": _float_or_none(row["sigma_dVinf_mm_s"]),
                    "Vinf_km_s": float(row["Vinf_km_s"]),
                    "delta_in_deg": float(row["delta_in_deg"]),
                    "delta_out_deg": float(row["delta_out_deg"]),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def write_anderson_summary(
    src: Path = DEFAULT_PUBLIC_ROWS,
    dst: Path = REPO_ROOT / "data" / "flyby" / "certificates" / "summary.csv",
) -> Path:
    rows = []
    for row in load_public_rows(src):
        prediction = anderson_mm_s(row["Vinf_km_s"], row["delta_in_deg"], row["delta_out_deg"])
        observed = float(row["dVinf_obs_mm_s"])
        rows.append(
            {
                **row,
                "anderson_mm_s": f"{prediction:.6f}",
                "anderson_residual_mm_s": f"{observed - prediction:.6f}",
                "oph_closed_target_mm_s": "0.0",
            }
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return dst


def closeflyby_certificate_from_public_row(
    row: dict[str, Any],
    *,
    source_manifest: dict[str, Any] | None = None,
    raw_manifest: dict[str, Any] | None = None,
    source_manifest_path: Path | None = DEFAULT_SOURCE_MANIFEST,
    raw_manifest_path: Path | None = None,
) -> dict[str, Any]:
    flyby_id = str(row["id"])
    prediction = anderson_mm_s(row["Vinf_km_s"], row["delta_in_deg"], row["delta_out_deg"])
    public_y = float(row["dVinf_obs_mm_s"])
    raw_receipt = _raw_od_receipt(raw_manifest, raw_manifest_path)
    cert = {
        "schema_version": "oph_closeflyby_certificate_v1",
        "tier": "Tier 1: CloseFlyByPublic(F)",
        "certificate_id": f"CloseFlyBy:{_slug(flyby_id)}",
        "flyby_id": flyby_id,
        "public_row": {
            "Vinf_km_s": row["Vinf_km_s"],
            "delta_in_deg": row["delta_in_deg"],
            "delta_out_deg": row["delta_out_deg"],
            "dVinf_obs_mm_s": public_y,
            "sigma_dVinf_mm_s": row["sigma_dVinf_mm_s"],
            "dVp_obs_mm_s": row["dVp_obs_mm_s"],
            "sigma_dVp_mm_s": row["sigma_dVp_mm_s"],
            "notes": row.get("notes", ""),
            "source_refs": _source_refs_for(flyby_id, source_manifest),
        },
        "anderson_comparator": {
            "K": K_MM_PER_KMS,
            "formula": "K*Vinf_km_s*(cos(delta_in_deg)-cos(delta_out_deg))",
            "prediction_mm_s": prediction,
            "residual_mm_s": public_y - prediction,
        },
        "raw_od_receipt": raw_receipt,
        "models": {
            "M_hist": {
                "status": "not_run",
                "purpose": "Reproduce the historical reduced orbit-determination residual.",
                "accepted_readout": "signed inbound/outbound asymptotic speed mismatch in mm/s",
                "required_outputs": ["A_hist_mm_s", "sigma_A_hist_mm_s", "historical_replay_pass"],
            },
            "M_full": {
                "status": "not_run",
                "purpose": "Replay the full conventional tracking record without target leakage.",
                "required_model_terms": [
                    "Earth gravity",
                    "Sun/Moon/planet perturbations",
                    "tides",
                    "relativity",
                    "station clocks and ramps",
                    "EOP",
                    "troposphere",
                    "ionosphere",
                    "spacecraft attitude",
                    "maneuvers",
                    "SRP/reflectivity/thermal recoil where available",
                ],
                "forbidden_closure_modes": ["explicit velocity impulse fitted only to erase the anomaly"],
                "required_outputs": ["A_full_mm_s", "sigma_A_full_mm_s", "full_replay_pass"],
            },
        },
        "results": {
            "A_hist_mm_s": None,
            "sigma_A_hist_mm_s": None,
            "A_full_mm_s": None,
            "sigma_A_full_mm_s": None,
            "A_proj_mm_s": None,
            "projection_decomposition": None,
        },
        "tests": {key: None for key in TEST_KEYS},
        "status": "SPEC_ONLY",
        "hashes": {
            "public_row_sha256": _json_sha256(row),
            "source_manifest_sha256": _file_sha256(source_manifest_path) if source_manifest_path else None,
            "raw_manifest_sha256": _file_sha256(raw_manifest_path) if raw_manifest_path else None,
        },
        "theorem_contract": {
            "object": "CloseFlyBy(F)",
            "closure_condition": [
                "A_hist approximates the public historical residual",
                "A_full is zero within the replay floor",
                "A_proj = A_hist - A_full approximates the public residual",
                "residual, covariance, station-holdout, and arc-holdout tests pass",
            ],
            "public_layer_claim": "certificate object and comparator are created; historical closure is not claimed",
        },
        "nonclaims": [
            "No historical flyby anomaly is solved by this public certificate layer.",
            "The Anderson comparator is recorded as a public empirical comparator, not as an OPH law.",
            "Synthetic projection tests validate code paths only and cannot close this historical row.",
            "No OPH new-force branch is promoted from an OD_REPLAY_PENDING row.",
        ],
        "next_required_artifacts": [
            "raw/reduced tracking files",
            "station, clock, EOP, troposphere, ionosphere, and weather calibration files",
            "SPICE/geometry and attitude products",
            "maneuver and non-gravitational force ledgers",
            "solve-for list, covariance, normal/design matrices, and OD code hash",
            "historical reduced replay, full conventional replay, projection artifact, and residual tests",
        ],
    }
    cert["status"] = classify_certificate(cert)
    cert["hashes"]["certificate_payload_sha256"] = _json_sha256(_without_certificate_hash(cert))
    return cert


def classify_certificate(
    cert: dict[str, Any],
    *,
    mission_floor_mm_s: float = 0.1,
    full_floor_mm_s: float = 0.1,
) -> str:
    if cert.get("synthetic_validation") is True:
        return "SYNTHETIC_PIPELINE_VALIDATED"

    public_row = cert.get("public_row", {})
    results = cert.get("results", {})
    tests = cert.get("tests", {})
    public_y = public_row.get("dVinf_obs_mm_s")
    sigma_public = public_row.get("sigma_dVinf_mm_s")
    a_full = results.get("A_full_mm_s")
    sigma_full = results.get("sigma_A_full_mm_s")
    if public_y is not None and a_full is not None:
        public_floor = max(abs(float(sigma_public or 0.0)), mission_floor_mm_s)
        full_floor = max(3.0 * abs(float(sigma_full or 0.0)), full_floor_mm_s)
        if abs(float(public_y)) <= public_floor and abs(float(a_full)) <= full_floor:
            return "SOLVED_NULL"

    raw_receipt = cert.get("raw_od_receipt", {})
    if not raw_receipt.get("tracking_files"):
        return "DATA_INCOMPLETE_NOT_SOLVED"

    if results.get("A_hist_mm_s") is None:
        return "OD_REPLAY_PENDING"

    if tests.get("historical_replay_pass") is not True:
        return "HISTORICAL_REPLAY_FAILED"

    if results.get("A_full_mm_s") is None:
        return "HISTORICAL_REPLAY_PASS"

    if tests.get("full_replay_pass") is not True:
        return "NOT_SOLVED_RESIDUAL"

    if results.get("A_proj_mm_s") is None:
        return "FULL_REPLAY_PASS"

    if tests.get("projection_pass") is True:
        return "SOLVED_PROJECTION_ARTIFACT"

    return "FULL_REPLAY_PASS_PROJECTION_UNRESOLVED"


def projection_operator(Jx: np.ndarray, Jn: np.ndarray, C: np.ndarray) -> np.ndarray:
    Jx = np.asarray(Jx, dtype=float)
    Jn = np.asarray(Jn, dtype=float)
    C = np.asarray(C, dtype=float)
    C_inv = np.linalg.inv(C)
    normal = Jx.T @ C_inv @ Jx
    rhs = Jx.T @ C_inv @ Jn
    return -np.linalg.solve(normal, rhs)


def projection_artifact(
    Jx: np.ndarray,
    Jn: np.ndarray,
    C: np.ndarray,
    delta_n: np.ndarray,
    L: np.ndarray,
) -> dict[str, Any]:
    P = projection_operator(Jx, Jn, C)
    delta_n = np.asarray(delta_n, dtype=float)
    L = np.asarray(L, dtype=float)
    delta_x = P @ delta_n
    return {
        "P_x_from_n": P,
        "delta_x": delta_x,
        "A_proj_mm_s": float(L @ delta_x),
    }


def synthetic_projection_validation() -> dict[str, Any]:
    Jx = np.array(
        [
            [1.0, 0.25],
            [1.0, -0.15],
            [1.0, 0.55],
            [1.0, -0.45],
            [1.0, 0.85],
            [1.0, -0.75],
        ],
        dtype=float,
    )
    Jn = np.array(
        [
            [0.15, 0.02],
            [0.05, -0.08],
            [0.22, 0.06],
            [-0.06, -0.14],
            [0.30, 0.12],
            [-0.10, -0.16],
        ],
        dtype=float,
    )
    C = np.diag([0.9, 1.2, 0.8, 1.0, 1.1, 0.95]) ** 2
    delta_n = np.array([0.42, -0.31], dtype=float)
    L = np.array([4.0, -7.5], dtype=float)
    artifact = projection_artifact(Jx, Jn, C, delta_n, L)
    P = artifact["P_x_from_n"]
    per_nuisance = []
    for index, nuisance_name in enumerate(("SRP_Cr", "station_bias")):
        delta_x_i = P[:, index] * delta_n[index]
        per_nuisance.append(
            {
                "nuisance": nuisance_name,
                "delta_n": float(delta_n[index]),
                "A_proj_component_mm_s": float(L @ delta_x_i),
            }
        )
    a_proj = float(artifact["A_proj_mm_s"])
    a_full = 0.0
    a_hist = a_proj
    certificate = {
        "schema_version": "oph_closeflyby_certificate_v1",
        "tier": "Tier 1 synthetic validation only",
        "certificate_id": "CloseFlyBy:synthetic_projection_fixture",
        "flyby_id": "SYNTHETIC_PROJECTION_FIXTURE",
        "synthetic_validation": True,
        "public_row": {
            "Vinf_km_s": 1.0,
            "delta_in_deg": 0.0,
            "delta_out_deg": 0.0,
            "dVinf_obs_mm_s": a_hist,
            "sigma_dVinf_mm_s": 1.0e-9,
            "dVp_obs_mm_s": a_hist,
            "sigma_dVp_mm_s": 1.0e-9,
            "source_refs": ["synthetic_unit_test_fixture"],
            "notes": "Synthetic fixture validates projection machinery only.",
        },
        "anderson_comparator": {
            "K": K_MM_PER_KMS,
            "formula": "not used for synthetic projection fixture",
            "prediction_mm_s": 0.0,
            "residual_mm_s": a_hist,
        },
        "raw_od_receipt": {
            **{key: [] for key in RAW_RECEIPT_ARRAY_KEYS},
            "tracking_files": [{"kind": "synthetic_observations", "status": "generated"}],
            "solve_for_list": ["asymptotic_speed_bias", "slope"],
            "covariance_file": "inline_synthetic_covariance",
            "od_code_hash": None,
            "receipt_status": "SYNTHETIC_ONLY_NOT_HISTORICAL_EVIDENCE",
        },
        "models": {
            "M_hist": {"status": "synthetic_reduced_model_omits_nuisance"},
            "M_full": {"status": "synthetic_full_model_includes_nuisance"},
        },
        "results": {
            "A_hist_mm_s": a_hist,
            "sigma_A_hist_mm_s": 0.0,
            "A_full_mm_s": a_full,
            "sigma_A_full_mm_s": 0.0,
            "A_proj_mm_s": a_proj,
            "projection_decomposition": {
                "components": per_nuisance,
                "component_sum_mm_s": float(sum(item["A_proj_component_mm_s"] for item in per_nuisance)),
                "closure_error_mm_s": float(a_hist - a_full - a_proj),
            },
        },
        "tests": {
            **{key: None for key in TEST_KEYS},
            "historical_replay_pass": True,
            "full_replay_pass": True,
            "projection_pass": abs(a_proj - (a_hist - a_full)) <= 1.0e-12,
        },
        "status": "SPEC_ONLY",
        "hashes": {},
        "nonclaims": [
            "Synthetic simulation validates code only.",
            "This certificate cannot close any historical flyby row.",
        ],
    }
    certificate["status"] = classify_certificate(certificate)
    certificate["hashes"] = {
        "fixture_sha256": _json_sha256(
            {
                "Jx": Jx.tolist(),
                "Jn": Jn.tolist(),
                "C": C.tolist(),
                "delta_n": delta_n.tolist(),
                "L": L.tolist(),
            }
        ),
        "certificate_payload_sha256": _json_sha256(certificate),
    }
    return {
        "Jx": Jx,
        "Jn": Jn,
        "C": C,
        "delta_n": delta_n,
        "L": L,
        "P_x_from_n": P,
        "delta_x": artifact["delta_x"],
        "A_hist_mm_s": a_hist,
        "A_full_mm_s": a_full,
        "A_proj_mm_s": a_proj,
        "projection_decomposition": per_nuisance,
        "certificate": certificate,
    }


def write_synthetic_closeflyby_validation(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = synthetic_projection_validation()
    serializable = _to_jsonable(validation)
    cert_path = out_dir / "synthetic_closeflyby_certificate.json"
    validation_path = out_dir / "synthetic_projection_validation.json"
    decomposition_path = out_dir / "projection_decomposition.csv"
    cert_path.write_text(json.dumps(serializable["certificate"], indent=2, sort_keys=True), encoding="utf-8")
    validation_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    with decomposition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["nuisance", "delta_n", "A_proj_component_mm_s"])
        writer.writeheader()
        writer.writerows(serializable["projection_decomposition"])
    return {
        "status": serializable["certificate"]["status"],
        "certificate": str(cert_path),
        "validation": str(validation_path),
        "projection_decomposition": str(decomposition_path),
        "A_hist_mm_s": serializable["A_hist_mm_s"],
        "A_full_mm_s": serializable["A_full_mm_s"],
        "A_proj_mm_s": serializable["A_proj_mm_s"],
    }


def write_closeflyby_public_certificates(
    out_dir: Path,
    *,
    public_rows_path: Path = DEFAULT_PUBLIC_ROWS,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    raw_root: Path = DEFAULT_RAW_ROOT,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load_yaml(source_manifest_path)
    certificate_paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for row in load_public_rows(public_rows_path):
        raw_manifest, raw_manifest_path = _load_raw_manifest(raw_root, str(row["id"]))
        cert = closeflyby_certificate_from_public_row(
            row,
            source_manifest=source_manifest,
            raw_manifest=raw_manifest,
            source_manifest_path=source_manifest_path,
            raw_manifest_path=raw_manifest_path,
        )
        cert_path = out_dir / f"{_filename_slug(str(row['id']))}.closeflyby.json"
        cert_path.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
        certificate_paths.append(cert_path)
        summary_rows.append(
            {
                "flyby_id": cert["flyby_id"],
                "status": cert["status"],
                "dVinf_obs_mm_s": cert["public_row"]["dVinf_obs_mm_s"],
                "sigma_dVinf_mm_s": cert["public_row"]["sigma_dVinf_mm_s"],
                "anderson_mm_s": f"{cert['anderson_comparator']['prediction_mm_s']:.6f}",
                "anderson_residual_mm_s": f"{cert['anderson_comparator']['residual_mm_s']:.6f}",
                "raw_receipt_status": cert["raw_od_receipt"]["receipt_status"],
                "missing_receipt_count": len(cert["raw_od_receipt"].get("missing_receipts", [])),
                "certificate_path": cert_path.name,
            }
        )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_path = out_dir / "closeflyby_summary.md"
    markdown_path.write_text(_render_markdown_summary(summary_rows), encoding="utf-8")

    if schema_path.exists():
        target_schema = out_dir / "closeflyby_schema.json"
        if schema_path.resolve() != target_schema.resolve():
            shutil.copyfile(schema_path, target_schema)

    report = {
        "schema": "oph_closeflyby_public_generation_report_v1",
        "status": "PUBLIC_CERTIFICATES_WRITTEN",
        "public_rows": str(public_rows_path),
        "source_manifest": str(source_manifest_path),
        "raw_root": str(raw_root),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "certificate_count": len(certificate_paths),
        "certificates": [str(path) for path in certificate_paths],
        "nonclaims": [
            "Generated certificates are not historical solutions.",
            "All non-synthetic historical rows remain pending until OD replay artifacts are attached.",
        ],
    }
    report_path = out_dir / "closeflyby_generation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def _raw_od_receipt(raw_manifest: dict[str, Any] | None, raw_manifest_path: Path | None) -> dict[str, Any]:
    receipt = {key: [] for key in RAW_RECEIPT_ARRAY_KEYS}
    receipt.update(
        {
            "solve_for_list": None,
            "covariance_file": None,
            "od_code_hash": None,
            "receipt_status": "PUBLIC_ROW_ONLY_REPLAY_PENDING",
            "missing_receipts": [],
        }
    )
    if not raw_manifest:
        return receipt

    required = raw_manifest.get("required_files", {}) or {}
    receipt["tracking_files"] = _required_items(required.get("tracking", []))
    calibration_items = _flatten_required(required.get("calibration", []))
    geometry_items = _flatten_required(required.get("geometry", []))
    model_items = _flatten_required(required.get("model", []))
    receipt["station_files"] = _matching_required(calibration_items, ("station", "media"))
    receipt["clock_ramp_time_files"] = _matching_required(calibration_items, ("clock", "time", "ramp"))
    receipt["eop_files"] = _matching_required(calibration_items + geometry_items, ("EOP", "eop"))
    receipt["tro_files"] = _matching_required(calibration_items, ("TRO", "troposphere", "media"))
    receipt["ion_files"] = _matching_required(calibration_items, ("ION", "ionosphere"))
    receipt["spice_files"] = _matching_required(geometry_items + model_items, ("SPICE", "spice", "nmlmodl", "Paramsum"))
    receipt["receipt_status"] = str(raw_manifest.get("status", "RAW_MANIFEST_REPLAY_PENDING"))
    receipt["missing_receipts"] = list(raw_manifest.get("missing_or_to_locate", []) or [])
    receipt["raw_manifest_path"] = str(raw_manifest_path) if raw_manifest_path else None
    receipt["raw_manifest_sha256"] = _file_sha256(raw_manifest_path) if raw_manifest_path else None
    return receipt


def _source_refs_for(flyby_id: str, source_manifest: dict[str, Any] | None) -> list[str]:
    refs: list[str] = []
    for source in (source_manifest or {}).get("sources", []) or []:
        source_id = str(source.get("id", ""))
        flyby_ids = source.get("flyby_ids")
        if flyby_ids is None or flyby_id in flyby_ids:
            refs.append(source_id)
    return refs


def _load_raw_manifest(raw_root: Path, flyby_id: str) -> tuple[dict[str, Any] | None, Path | None]:
    direct_path = raw_root / _slug(flyby_id) / "manifest.yaml"
    if direct_path.exists():
        return _load_yaml(direct_path), direct_path
    null_path = raw_root / "null_controls" / "manifest.yaml"
    if null_path.exists():
        manifest = _load_yaml(null_path)
        if flyby_id in (manifest.get("flyby_ids", []) or []):
            return manifest, null_path
    return None, None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping in {path}")
    return loaded


def _required_items(values: Any) -> list[dict[str, str]]:
    return [{"required_type": item, "status": "manifest_only_not_bundled"} for item in _flatten_required(values)]


def _matching_required(values: list[str], needles: tuple[str, ...]) -> list[dict[str, str]]:
    matched = []
    for value in values:
        lower_value = value.lower()
        if any(needle.lower() in lower_value for needle in needles):
            matched.append({"required_type": value, "status": "manifest_only_not_bundled"})
    return matched


def _flatten_required(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        out: list[str] = []
        for value in values:
            out.extend(_flatten_required(value))
        return out
    if isinstance(values, dict):
        out = []
        for value in values.values():
            out.extend(_flatten_required(value))
        return out
    return [str(values)]


def _render_markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CloseFlyBy(F) public certificates",
        "",
        "These rows are generated from public comparator data and raw-data availability manifests.",
        "They are not solved historical certificates until OD replay fills A_hist, A_full, A_proj, and residual tests.",
        "",
        "| Flyby | Status | Public dVinf mm/s | Anderson mm/s | Residual mm/s | Raw receipt | Missing receipts |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {flyby_id} | {status} | {dVinf_obs_mm_s} | {anderson_mm_s} | "
            "{anderson_residual_mm_s} | {raw_receipt_status} | {missing_receipt_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Closure rule: a non-null row can become SOLVED_PROJECTION_ARTIFACT only after real-data OD replay shows",
            "A_hist approximately equals the public residual, A_full is zero within the replay floor,",
            "A_proj equals A_hist - A_full and matches the public residual, and residual/covariance/holdout tests pass.",
            "",
            "Synthetic projection fixtures validate this machinery only; they do not close historical rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _filename_slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_")


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(_to_jsonable(data), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _without_certificate_hash(cert: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(_to_jsonable(cert)))
    clone.setdefault("hashes", {}).pop("certificate_payload_sha256", None)
    return clone


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
