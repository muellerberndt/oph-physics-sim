from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import COSMOLOGY_PERTURBATION_RECEIPT, QUANTITATIVE_BRANCH, with_claim_metadata
from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.finite_certificates import (
    ANOMALY_CURRENT_CONSERVATION_RECEIPT,
    C_SI,
    HBAR_SI,
    SOURCE_LOCALIZATION_SATURATION_RECEIPT,
)


DEFAULT_SOURCE_PATH = Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/inflation/3/comms.md")

CERTIFICATE_TYPES = (
    "scalar_release",
    "edge_center",
    "homogeneous_anomaly",
    "parent_collar",
    "repair_matrix",
    "boltzmann_handoff",
)

DEFAULT_FILENAMES = {
    "scalar_release": "scalar_release_certificate.json",
    "edge_center": "edge_center_certificate.json",
    "homogeneous_anomaly": "homogeneous_anomaly_certificate.json",
    "parent_collar": "parent_collar_certificate.json",
    "repair_matrix": "repair_matrix_certificate.json",
    "boltzmann_handoff": "boltzmann_handoff_certificate.json",
}

FORBIDDEN_DATA_USE_KEYS = (
    "cmb_likelihood",
    "planck_likelihood",
    "bao_likelihood",
    "desi_likelihood",
    "weak_lensing_likelihood",
    "sparc_likelihood",
    "cluster_likelihood",
    "supernova_likelihood",
    "rsd_likelihood",
    "observed_A_s",
    "observed_sigma8",
    "observed_S8",
)


def inflation_certificate_bundle_report(
    cert_dir: Path | None = None,
    *,
    source_path: Path | None = DEFAULT_SOURCE_PATH,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    """Validate inflation/CMB certificate artifacts when present.

    Certificates are finite simulation outputs. Missing artifacts therefore keep
    the physical prediction gates closed; this report is a contract/readiness
    layer, not a source of toy numeric certificates.
    """

    certs = _load_certificates(cert_dir)
    validations = {
        cert_type: validate_certificate(certs.get(cert_type), cert_type=cert_type, p_value=p_value)
        for cert_type in CERTIFICATE_TYPES
    }
    no_data_use = _bundle_no_data_use_report(certs)
    passed = [name for name, receipt in validations.items() if receipt.get("validator_receipt")]
    report = {
        "mode": "oph_inflation_certificate_bundle_v0",
        "source_files": _source_status(source_path),
        "cert_dir": str(cert_dir) if cert_dir is not None else None,
        "expected_certificate_types": list(CERTIFICATE_TYPES),
        "expected_filenames": dict(DEFAULT_FILENAMES),
        "certificates_found": sorted(certs),
        "certificate_validations": validations,
        "certificate_summary": {
            "expected_count": len(CERTIFICATE_TYPES),
            "found_count": len(certs),
            "passed_count": len(passed),
            "passed_types": passed,
            "missing_types": [name for name in CERTIFICATE_TYPES if name not in certs],
            "failed_types": [
                name for name, receipt in validations.items() if name in certs and not receipt.get("validator_receipt")
            ],
        },
        "schema_contracts": schema_contracts(),
        "no_data_use_manifest": no_data_use,
        "readiness_gates": {
            "scalar_release_certificate": bool(
                (validations["scalar_release"].get("computed_outputs") or {}).get(
                    "SCALAR_RELEASE_AMPLITUDE_CERTIFICATE", False
                )
            ),
            "edge_center_certificate": bool(validations["edge_center"].get("validator_receipt")),
            "homogeneous_anomaly_certificate": bool(validations["homogeneous_anomaly"].get("validator_receipt")),
            "parent_collar_kernel_certificate": bool(validations["parent_collar"].get("validator_receipt")),
            "repair_matrix_certificate": bool(validations["repair_matrix"].get("validator_receipt")),
            "boltzmann_handoff_certificate": bool(validations["boltzmann_handoff"].get("validator_receipt")),
            "no_data_use_firewall": bool(no_data_use.get("no_data_use_receipt")),
        },
        "derived_outputs": _derived_outputs(validations),
        "measurement_comparable_now": False,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Inflation/CMB finite-certificate contract and validator report. Certificates must be emitted "
            "by finite OPH screen/collar evaluators before CMB/BAO/lensing/SPARC/cluster/SN/RSD likelihoods "
            "are consulted. Missing or failed certificates keep A_q, the screen-to-primordial lift, "
            "A_zeta, Q_A, B_A(k,a), Gamma_rec(k,a), and Boltzmann handoff gates closed."
        ),
    }
    report["inflation_certificate_stack_ready"] = bool(
        all(report["readiness_gates"].values()) and len(passed) == len(CERTIFICATE_TYPES)
    )
    return with_claim_metadata(
        report,
        claim_level=QUANTITATIVE_BRANCH,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_inflation_certificate_bundle",
        fit_objective="finite_certificate_readiness_audit",
    )


def write_inflation_certificate_bundle_report(
    cert_dir: Path | None,
    out_dir: Path,
    *,
    source_path: Path | None = DEFAULT_SOURCE_PATH,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    report = inflation_certificate_bundle_report(cert_dir, source_path=source_path, p_value=p_value)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inflation_certificate_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "inflation_certificate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "inflation_certificate_rows.csv", _validation_rows(report))
    schema_dir = out / "schemas"
    schema_dir.mkdir(exist_ok=True)
    for name, schema in report["schema_contracts"].items():
        (schema_dir / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    template_dir = out / "templates"
    template_dir.mkdir(exist_ok=True)
    for name, template in certificate_templates(p_value=p_value).items():
        (template_dir / f"{name}_certificate.template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
    return report


def emit_scalar_release_certificate_from_collar_run(
    run_dir: Path,
    out_dir: Path,
    *,
    kappa_rel: float = 1.0,
    source_path: Path | None = DEFAULT_SOURCE_PATH,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    """Emit a scalar-release certificate from cached collar CMI receipts.

    This is intentionally a proxy certificate. Current collar reports persist
    diagonal empirical CMI rows, not the full noncommutative cap algebra needed
    for a paper-grade scalar-release proof. Persisting this artifact still lets
    downstream cosmology tooling consume exactly what the finite run established.
    """

    root = Path(run_dir)
    out = Path(out_dir)
    collar_path = root / "collar_markov_report.json"
    manifest_path = root / "manifest.json"
    bw_path = root / "bw_state_derived_report.json"
    collar = _read_json(collar_path)
    if not collar:
        raise FileNotFoundError(f"missing or invalid collar_markov_report.json in {root}")
    rows = collar.get("rows") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"collar report has no rows: {collar_path}")
    packets = []
    for index, row in enumerate(rows):
        cmi = _float_or_none(row.get("epsilon_cmi"))
        if cmi is None:
            continue
        packets.append(
            {
                "packet_id": f"cap_{row.get('cap_id', index)}",
                "scalar_visible": True,
                "cmi": float(cmi),
                "source_cap_id": row.get("cap_id"),
                "theta0": row.get("theta0"),
                "collar_width": row.get("collar_width"),
                "inside_count": row.get("inside_count"),
                "collar_count": row.get("collar_count"),
                "outside_count": row.get("outside_count"),
                "sample_count": row.get("sample_count"),
                "packet_alphabet_size": row.get("packet_alphabet_size"),
                "sector_conditioned_cmi": row.get("sector_conditioned_cmi", {}),
            }
        )
    if not packets:
        raise ValueError(f"collar report has no positive numeric epsilon_cmi rows: {collar_path}")

    source_hashes = {"collar_markov_report": _sha256_file(collar_path)}
    if manifest_path.exists():
        source_hashes["manifest"] = _sha256_file(manifest_path)
    if bw_path.exists():
        source_hashes["bw_state_derived_report"] = _sha256_file(bw_path)

    manifest = _read_json(manifest_path)
    certificate = {
        "id": f"scalar_release_from_{manifest.get('run_id') or root.name}",
        "type": "scalar_release",
        "certificate_tier": "diagonal_collar_markov_proxy",
        "source_run_dir": str(root),
        "source_hashes": source_hashes,
        "release_packets": packets,
        "scalar_readout_normalization": {
            "kappa_rel": float(kappa_rel),
            "source": "user_supplied_proxy_normalization_for_finite_collar_receipt",
        },
        "collar_summary": {
            "mode": collar.get("mode"),
            "cap_count": collar.get("cap_count"),
            "median_epsilon_cmi": collar.get("median_epsilon_cmi"),
            "mean_epsilon_cmi": collar.get("mean_epsilon_cmi"),
            "p90_epsilon_cmi": collar.get("p90_epsilon_cmi"),
        },
        "no_data_use_manifest": _template_no_data_use(),
        "claim_boundary": (
            "Proxy scalar-release certificate emitted from diagonal empirical cap/collar CMI rows. "
            "It is a finite screen receipt for a support-visible scalar-release lane, not a "
            "paper-grade noncommutative scalar-release proof and not a physical CMB prediction."
        ),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / DEFAULT_FILENAMES["scalar_release"]).write_text(
        json.dumps(certificate, indent=2, default=str),
        encoding="utf-8",
    )
    report = write_inflation_certificate_bundle_report(out, out, source_path=source_path, p_value=p_value)
    report["emitted_certificate"] = {
        "path": str(out / DEFAULT_FILENAMES["scalar_release"]),
        "certificate_tier": certificate["certificate_tier"],
        "source_run_dir": str(root),
        "packet_count": len(packets),
    }
    (out / "inflation_certificate_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "inflation_certificate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def emit_edge_center_certificate(
    out_dir: Path,
    *,
    p_value: float = P_STAR,
    source_path: Path | None = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    """Emit the finite edge-center certificate for the P/48 tilt target."""

    out = Path(out_dir)
    p = float(p_value)
    certificate = {
        "id": "edge_center_p48_screen_microphysics",
        "type": "edge_center",
        "certificate_tier": "paper_formula_finite_edge_center",
        "P": p,
        "edge_center_basis": ["scalar_event", "z6_reserve", "sector_local_generator"],
        "scalar_event_diagonal": [1.0, 0.0, 0.0],
        "z6_reserve_diagonal": [0.0, 1.0, 0.0],
        "sector_local_generators": [
            {"name": "support_visible_sector_local_generator", "diagonal": [0.0, 0.0, 1.0]}
        ],
        "no_data_use_manifest": _template_no_data_use(),
        "claim_boundary": (
            "Finite edge-center readout for the paper-side P/48 tilt target. This supplies the "
            "screen-microphysics n_s target without observational fitting; it is not a finite "
            "lattice derivation of the scalar amplitude or a physical CMB prediction by itself."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / DEFAULT_FILENAMES["edge_center"]
    path.write_text(json.dumps(certificate, indent=2, default=str), encoding="utf-8")
    report = write_inflation_certificate_bundle_report(out, out, source_path=source_path, p_value=p)
    report["emitted_certificate"] = {
        "path": str(path),
        "certificate_tier": certificate["certificate_tier"],
        "P": p,
    }
    (out / "inflation_certificate_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "inflation_certificate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def validate_certificate(
    certificate: dict[str, Any] | None,
    *,
    cert_type: str | None = None,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    if not certificate:
        return _missing_receipt(cert_type or "unknown")
    declared_type = str(certificate.get("type") or cert_type or "")
    expected_type = cert_type or declared_type
    if declared_type and expected_type and declared_type != expected_type:
        return {
            "type": expected_type,
            "present": True,
            "validator_receipt": False,
            "reason": f"type mismatch: declared {declared_type}, expected {expected_type}",
        }
    if expected_type == "scalar_release":
        return _validate_scalar_release(certificate)
    if expected_type == "edge_center":
        return _validate_edge_center(certificate, p_value=p_value)
    if expected_type == "homogeneous_anomaly":
        return _validate_homogeneous_anomaly(certificate)
    if expected_type == "parent_collar":
        return _validate_parent_collar(certificate)
    if expected_type == "repair_matrix":
        return _validate_repair_matrix(certificate)
    if expected_type == "boltzmann_handoff":
        return _validate_boltzmann_handoff(certificate)
    return {"type": expected_type, "present": True, "validator_receipt": False, "reason": "unknown certificate type"}


def schema_contracts() -> dict[str, dict[str, Any]]:
    return {
        "scalar_release": {
            "required": ["id", "type", "release_packets", "scalar_readout_normalization", "no_data_use_manifest"],
            "packet_required": ["packet_id", "scalar_visible"],
            "packet_required_one_of": ["entropy", "cmi"],
            "entropy_required": ["AB", "BD", "B", "ABD"],
            "computed_outputs": [
                "epsilon_star",
                "kappa_rel",
                "A_q_cmi_upper_bound",
                "A_q_energy",
                "A_zeta",
                "SCALAR_RELEASE_AMPLITUDE_CERTIFICATE",
            ],
        },
        "edge_center": {
            "required": [
                "id",
                "type",
                "edge_center_basis",
                "scalar_event_diagonal",
                "z6_reserve_diagonal",
                "sector_local_generators",
                "no_data_use_manifest",
            ],
            "computed_outputs": ["lambda_collar", "lambda_scalar", "theta_OPH", "n_s"],
        },
        "homogeneous_anomaly": {
            "required": ["id", "type", "refinement_levels", "no_data_use_manifest"],
            "level_required": ["level", "a", "V_com", "ell_r", "collars"],
            "collar_required": ["weight", "cmi"],
            "computed_outputs": ["rho_A_r", "Q_A_r", "cauchy_convergence"],
        },
        "parent_collar": {
            "required": ["id", "type", "kernel_rows", "no_data_use_manifest"],
            "row_required": ["k", "a", "rho_b_bar", "rho_A_bar", "d_rho_A_eq_d_delta_b"],
            "computed_outputs": ["K_A_rho", "B_A"],
        },
        "repair_matrix": {
            "required": ["id", "type", "transition_matrices", "no_data_use_manifest"],
            "matrix_required": ["k", "a", "delta_eta", "stationary_distribution", "K"],
            "computed_outputs": ["lambda_2_abs", "Gamma_rec"],
        },
        "boltzmann_handoff": {
            "required": ["id", "type", "required_outputs", "certificate_references", "no_data_use_manifest"],
            "required_outputs": [
                "background_A",
                "perturbation_A_grid",
                "B_A_grid",
                "Gamma_rec_grid",
                "primordial",
                "neutrino_branch",
                "solver_manifest",
            ],
            "computed_outputs": ["handoff_ready"],
        },
    }


def certificate_templates(*, p_value: float = P_STAR) -> dict[str, dict[str, Any]]:
    return {
        "scalar_release": {
            "id": "scalar_release_r0001",
            "type": "scalar_release",
            "release_packets": [
                {
                    "packet_id": "example",
                    "scalar_visible": True,
                    "entropy": {"AB": 0.0, "BD": 0.0, "B": 0.0, "ABD": 0.0},
                }
            ],
            "scalar_readout_normalization": {"kappa_rel": None},
            "scalar_energy_budget": None,
            "physical_mode_count": None,
            "bound_saturation_claimed": False,
            "DEFAULT_A_S_used": False,
            "CMB_homogeneity_target_used": False,
            "Sachs_Wolfe_conversion_used": False,
            "A_zeta": None,
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
        "edge_center": {
            "id": "edge_center_r0001",
            "type": "edge_center",
            "P": float(p_value),
            "edge_center_basis": ["sector_0", "sector_1"],
            "scalar_event_diagonal": [1, 0],
            "z6_reserve_diagonal": [1, 0],
            "sector_local_generators": [{"name": "example", "diagonal": [0, 1]}],
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
        "homogeneous_anomaly": {
            "id": "homogeneous_anomaly_r0001",
            "type": "homogeneous_anomaly",
            "refinement_levels": [],
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
        "parent_collar": {
            "id": "parent_collar_r0001",
            "type": "parent_collar",
            "kernel_rows": [],
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
        "repair_matrix": {
            "id": "repair_matrix_r0001",
            "type": "repair_matrix",
            "transition_matrices": [],
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
        "boltzmann_handoff": {
            "id": "boltzmann_handoff_r0001",
            "type": "boltzmann_handoff",
            "required_outputs": {},
            "certificate_references": [],
            "no_data_use_manifest": _template_no_data_use(),
            "template_only": True,
        },
    }


def _validate_scalar_release(certificate: dict[str, Any]) -> dict[str, Any]:
    packets = certificate.get("release_packets") or []
    kappa = _nested(certificate, "scalar_readout_normalization", "kappa_rel")
    forbidden_flags = [
        name
        for name in (
            "bound_saturation_claimed",
            "DEFAULT_A_S_used",
            "CMB_homogeneity_target_used",
            "Sachs_Wolfe_conversion_used",
        )
        if bool(certificate.get(name, False))
    ]
    if certificate.get("A_zeta") is not None:
        forbidden_flags.append("A_zeta_present_before_lift_receipt")
    if forbidden_flags:
        return _failed_receipt("scalar_release", f"forbidden scalar-amplitude shortcut: {forbidden_flags}")
    if not isinstance(packets, list) or not packets:
        return _failed_receipt("scalar_release", "missing release_packets")
    kappa_value = _float_or_none(kappa)
    if kappa_value is None or kappa_value <= 0.0:
        return _failed_receipt("scalar_release", "missing positive scalar_readout_normalization.kappa_rel")
    rows = []
    for packet in packets:
        entropy = packet.get("entropy") or {}
        cmi = _float_or_none(packet.get("cmi"))
        if cmi is None:
            values = [_float_or_none(entropy.get(key)) for key in ("AB", "BD", "B", "ABD")]
            if any(value is None for value in values):
                continue
            cmi = float(values[0]) + float(values[1]) - float(values[2]) - float(values[3])
        rows.append(
            {
                "packet_id": packet.get("packet_id"),
                "scalar_visible": bool(packet.get("scalar_visible", False)),
                "cmi": float(cmi),
            }
        )
    positives = [row["cmi"] for row in rows if row["scalar_visible"] and row["cmi"] > 0.0 and np.isfinite(row["cmi"])]
    if not positives:
        return _failed_receipt("scalar_release", "no positive scalar-visible CMI packet", rows=rows)
    epsilon = float(min(positives))
    cmi_upper_bound = float(4.0 * math.log(2.0) * kappa_value * epsilon)
    energy_budget = _float_or_none(certificate.get("scalar_energy_budget"))
    physical_mode_count = _float_or_none(certificate.get("physical_mode_count"))
    a_q_energy = None
    amplitude_receipt = False
    if energy_budget is not None and physical_mode_count is not None and energy_budget >= 0.0 and physical_mode_count > 0.0:
        a_q_energy = float(energy_budget / physical_mode_count)
        amplitude_receipt = True
    return _receipt(
        "scalar_release",
        computed_outputs={
            "epsilon_star": epsilon,
            "kappa_rel": kappa_value,
            "A_q_cmi_upper_bound": cmi_upper_bound,
            "A_q_energy": a_q_energy,
            "scalar_energy_budget": energy_budget,
            "physical_mode_count": physical_mode_count,
            "SCALAR_RELEASE_AMPLITUDE_CERTIFICATE": amplitude_receipt,
            "bound_saturation_claimed": False,
            "DEFAULT_A_S_used": False,
            "CMB_homogeneity_target_used": False,
            "Sachs_Wolfe_conversion_used": False,
            "A_zeta": None,
            "claim_boundary": (
                "CMI supplies an upper bound on scalar release observables. A_q is derived only from "
                "finite scalar release energy divided by physical mode count; A_zeta remains null until "
                "a screen-to-primordial lift receipt passes."
            ),
        },
        rows=rows,
        no_data_use=_no_data_use_ok(certificate),
    )


def _validate_edge_center(certificate: dict[str, Any], *, p_value: float) -> dict[str, Any]:
    p = _float_or_none(certificate.get("P"))
    p = float(p_value if p is None else p)
    scalar = _as_array(certificate.get("scalar_event_diagonal"))
    z6 = _as_array(certificate.get("z6_reserve_diagonal"))
    basis = certificate.get("edge_center_basis") or []
    if scalar is None or z6 is None or scalar.shape != z6.shape or len(basis) != int(scalar.size):
        return _failed_receipt("edge_center", "edge-center basis and diagonal events must have matching sizes")
    generators = certificate.get("sector_local_generators") or []
    diagonal_generators = []
    for item in generators:
        diag = _as_array(item.get("diagonal") if isinstance(item, dict) else None)
        if diag is None or diag.shape != scalar.shape:
            return _failed_receipt("edge_center", "all sector-local generators must be diagonal arrays in this finite certificate")
        diagonal_generators.append(str(item.get("name", "unnamed")))
    outputs = {
        "P": p,
        "lambda_collar": float(math.exp(-p / 24.0)),
        "lambda_scalar": float(math.exp(-p / 48.0)),
        "theta_OPH": float(p / 48.0),
        "n_s": float(1.0 - p / 48.0),
        "sector_local_generator_count": len(diagonal_generators),
        "scalar_z6_commutator_norm": 0.0,
    }
    return _receipt("edge_center", computed_outputs=outputs, no_data_use=_no_data_use_ok(certificate))


def _validate_homogeneous_anomaly(certificate: dict[str, Any]) -> dict[str, Any]:
    levels = certificate.get("refinement_levels") or []
    if not isinstance(levels, list) or not levels:
        return _failed_receipt("homogeneous_anomaly", "missing refinement_levels")
    if not bool(certificate.get(SOURCE_LOCALIZATION_SATURATION_RECEIPT, False)):
        return _failed_receipt("homogeneous_anomaly", "missing SOURCE_LOCALIZATION_SATURATION_RECEIPT")
    rows = []
    for level in levels:
        a = _float_or_none(level.get("a"))
        v_com = _float_or_none(level.get("V_com"))
        ell = _float_or_none(level.get("proper_ell_m", level.get("ell_r")))
        c_light = _float_or_none(level.get("c_si")) or C_SI
        collars = level.get("collars") or []
        if a is None or v_com is None or ell is None or ell <= 0.0 or not collars:
            continue
        weights = np.asarray([_float_or(row.get("weight"), 0.0) for row in collars], dtype=float)
        cmi = np.asarray([_float_or(row.get("cmi_diagnostic_nats", row.get("cmi")), 0.0) for row in collars], dtype=float)
        charge = np.asarray([_float_or(row.get("modular_source_charge_nats"), math.nan) for row in collars], dtype=float)
        if np.sum(weights) <= 0.0:
            continue
        if not np.all(np.isfinite(charge)) or np.any(charge < 0.0):
            continue
        weighted_cmi = float(np.sum(weights * cmi) / np.sum(weights))
        weighted_charge = float(np.sum(weights * charge) / np.sum(weights))
        rho_a = float((15.0 * HBAR_SI / (8.0 * math.pi * math.pi * c_light * ell**4)) * weighted_charge)
        q_a = float((a**3) * v_com * rho_a) if certificate.get(ANOMALY_CURRENT_CONSERVATION_RECEIPT, False) else None
        rows.append(
            {
                "level": level.get("level"),
                "a": a,
                "V_com": v_com,
                "proper_ell_m": ell,
                "weighted_cmi_diagnostic_nats": weighted_cmi,
                "weighted_modular_source_charge_nats": weighted_charge,
                "rho_A_r": rho_a,
                "Q_A_r": q_a,
            }
        )
    if not rows:
        return _failed_receipt("homogeneous_anomaly", "no valid refinement rows")
    if not certificate.get(ANOMALY_CURRENT_CONSERVATION_RECEIPT, False):
        return _receipt(
            "homogeneous_anomaly",
            validator_receipt=False,
            reason="missing ANOMALY_CURRENT_CONSERVATION_RECEIPT for conserved Q_A refinement",
            computed_outputs={"Q_A_last": None, "row_count": len(rows)},
            rows=rows,
            no_data_use=_no_data_use_ok(certificate),
        )
    q_values = [row["Q_A_r"] for row in rows if row["Q_A_r"] is not None]
    diffs = [abs(q_values[index + 1] - q_values[index]) for index in range(len(q_values) - 1)]
    tolerances = [float(level.get("epsilon_r", math.inf)) for level in levels[1:]]
    cauchy = bool(not diffs or all(diff <= tol for diff, tol in zip(diffs, tolerances)))
    return _receipt(
        "homogeneous_anomaly",
        validator_receipt=cauchy,
        reason=None if cauchy else "Q_A refinement differences exceed supplied epsilon_r",
        computed_outputs={"Q_A_last": float(q_values[-1]), "cauchy_convergence": cauchy, "max_Q_A_delta": max(diffs) if diffs else 0.0},
        rows=rows,
        no_data_use=_no_data_use_ok(certificate),
    )


def _validate_parent_collar(certificate: dict[str, Any]) -> dict[str, Any]:
    rows_in = certificate.get("kernel_rows") or []
    if not isinstance(rows_in, list) or not rows_in:
        return _failed_receipt("parent_collar", "missing kernel_rows")
    rows = []
    for row in rows_in:
        rho_b = _float_or_none(row.get("rho_b_bar"))
        rho_a = _float_or_none(row.get("rho_A_bar"))
        derivative = _float_or_none(row.get("d_rho_A_eq_d_delta_b"))
        if rho_b is None or rho_a is None or derivative is None or rho_b <= 0.0 or rho_a <= 0.0:
            continue
        k_a_rho = float(derivative / rho_b)
        b_a = float((rho_b / rho_a) * k_a_rho)
        rows.append({"k": row.get("k"), "a": row.get("a"), "K_A_rho": k_a_rho, "B_A": b_a})
    if not rows:
        return _failed_receipt("parent_collar", "no valid kernel rows")
    return _receipt(
        "parent_collar",
        computed_outputs={"row_count": len(rows), "mean_B_A": _mean(row["B_A"] for row in rows)},
        rows=rows,
        no_data_use=_no_data_use_ok(certificate),
    )


def _validate_repair_matrix(certificate: dict[str, Any]) -> dict[str, Any]:
    matrices = certificate.get("transition_matrices") or []
    if not isinstance(matrices, list) or not matrices:
        return _failed_receipt("repair_matrix", "missing transition_matrices")
    rows = []
    max_balance_error = 0.0
    max_row_sum_error = 0.0
    for item in matrices:
        matrix = np.asarray(item.get("K"), dtype=float)
        pi = np.asarray(item.get("stationary_distribution"), dtype=float)
        delta_eta = _float_or_none(item.get("delta_eta"))
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or pi.shape != (matrix.shape[0],):
            return _failed_receipt("repair_matrix", "K must be square and pi must match its size")
        if delta_eta is None or delta_eta <= 0.0:
            return _failed_receipt("repair_matrix", "delta_eta must be positive")
        if np.any(matrix < -1.0e-12) or np.any(pi < -1.0e-12) or np.sum(pi) <= 0.0:
            return _failed_receipt("repair_matrix", "K and pi must be nonnegative")
        pi = pi / np.sum(pi)
        row_sum_error = float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0)))
        balance_error = float(np.max(np.abs(pi[:, None] * matrix - pi[None, :] * matrix.T)))
        max_row_sum_error = max(max_row_sum_error, row_sum_error)
        max_balance_error = max(max_balance_error, balance_error)
        eig = np.linalg.eigvals(matrix)
        sorted_abs = sorted((float(abs(value)) for value in eig), reverse=True)
        lambda_2 = sorted_abs[1] if len(sorted_abs) > 1 else 0.0
        gamma = float(-math.log(max(lambda_2, 1.0e-300)) / delta_eta)
        rows.append(
            {
                "k": item.get("k"),
                "a": item.get("a"),
                "lambda_2_abs": lambda_2,
                "Gamma_rec": gamma,
                "row_sum_error": row_sum_error,
                "detailed_balance_error": balance_error,
            }
        )
    valid = bool(max_row_sum_error < 1.0e-9 and max_balance_error < 1.0e-9)
    return _receipt(
        "repair_matrix",
        validator_receipt=valid,
        reason=None if valid else "transition matrix stochasticity or detailed balance failed",
        computed_outputs={
            "row_count": len(rows),
            "mean_Gamma_rec": _mean(row["Gamma_rec"] for row in rows),
            "max_row_sum_error": max_row_sum_error,
            "max_detailed_balance_error": max_balance_error,
        },
        rows=rows,
        no_data_use=_no_data_use_ok(certificate),
    )


def _validate_boltzmann_handoff(certificate: dict[str, Any]) -> dict[str, Any]:
    required = schema_contracts()["boltzmann_handoff"]["required_outputs"]
    outputs = certificate.get("required_outputs") or {}
    present = {name: bool(outputs.get(name)) for name in required}
    refs = certificate.get("certificate_references") or []
    refs_ok = bool(isinstance(refs, list) and len(refs) >= 4)
    ready = bool(all(present.values()) and refs_ok)
    return _receipt(
        "boltzmann_handoff",
        validator_receipt=ready,
        reason=None if ready else "missing required outputs or certificate references",
        computed_outputs={"handoff_ready": ready, "present_outputs": present, "reference_count": len(refs) if isinstance(refs, list) else 0},
        no_data_use=_no_data_use_ok(certificate),
    )


def _load_certificates(cert_dir: Path | None) -> dict[str, dict[str, Any]]:
    if cert_dir is None:
        return {}
    root = Path(cert_dir)
    if not root.exists():
        return {}
    certificates: dict[str, dict[str, Any]] = {}
    for cert_type, filename in DEFAULT_FILENAMES.items():
        path = root / filename
        data = _read_json(path)
        if data:
            data.setdefault("_source_path", str(path))
            data.setdefault("_source_sha256", _sha256_file(path))
            certificates[cert_type] = data
    for path in root.glob("**/*certificate*.json"):
        if "templates" in path.parts or "schemas" in path.parts:
            continue
        data = _read_json(path)
        if data.get("template_only"):
            continue
        declared = data.get("type") if isinstance(data, dict) else None
        if declared in CERTIFICATE_TYPES and declared not in certificates:
            data.setdefault("_source_path", str(path))
            data.setdefault("_source_sha256", _sha256_file(path))
            certificates[str(declared)] = data
    return certificates


def _bundle_no_data_use_report(certs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for cert_type in CERTIFICATE_TYPES:
        cert = certs.get(cert_type)
        if not cert:
            rows.append({"type": cert_type, "present": False, "no_data_use_ok": False, "reason": "missing certificate"})
            continue
        status = _no_data_use_ok(cert)
        rows.append({"type": cert_type, **status})
    present_rows = [row for row in rows if row["present"]]
    return {
        "rows": rows,
        "no_data_use_receipt": bool(present_rows and all(row["no_data_use_ok"] for row in present_rows)),
        "forbidden_keys": list(FORBIDDEN_DATA_USE_KEYS),
        "claim_boundary": (
            "This checks certificate manifests only. It does not prove the generator code never read "
            "external measurements unless the source hashes and build environment are also audited."
        ),
    }


def _no_data_use_ok(certificate: dict[str, Any]) -> dict[str, Any]:
    manifest = certificate.get("no_data_use_manifest") or {}
    if not isinstance(manifest, dict):
        return {"present": True, "no_data_use_ok": False, "reason": "no_data_use_manifest is not an object"}
    used = set(str(item) for item in manifest.get("used_data") or [])
    forbidden = set(str(item) for item in manifest.get("forbidden_data") or [])
    explicit = manifest.get("observational_likelihoods_used")
    disallowed_used = sorted(used.intersection(FORBIDDEN_DATA_USE_KEYS))
    forbidden_declared = all(key in forbidden for key in FORBIDDEN_DATA_USE_KEYS)
    ok = bool(explicit is False and not disallowed_used and forbidden_declared)
    reason = None
    if explicit is not False:
        reason = "observational_likelihoods_used must be false"
    elif disallowed_used:
        reason = f"forbidden data used: {disallowed_used}"
    elif not forbidden_declared:
        reason = "forbidden_data list does not declare the full firewall"
    return {"present": True, "no_data_use_ok": ok, "reason": reason}


def _derived_outputs(validations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name, receipt in validations.items():
        if receipt.get("validator_receipt"):
            outputs[name] = receipt.get("computed_outputs", {})
    return outputs


def _receipt(
    cert_type: str,
    *,
    validator_receipt: bool = True,
    reason: str | None = None,
    computed_outputs: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    no_data_use: dict[str, Any] | None = None,
) -> dict[str, Any]:
    no_data = no_data_use or {"no_data_use_ok": False, "reason": "missing no-data-use status"}
    receipt = bool(validator_receipt and no_data.get("no_data_use_ok", False))
    return {
        "type": cert_type,
        "present": True,
        "validator_receipt": receipt,
        "structural_validator_receipt": bool(validator_receipt),
        "no_data_use_ok": bool(no_data.get("no_data_use_ok", False)),
        "reason": reason or no_data.get("reason"),
        "computed_outputs": computed_outputs or {},
        "rows": rows or [],
    }


def _missing_receipt(cert_type: str) -> dict[str, Any]:
    return {"type": cert_type, "present": False, "validator_receipt": False, "reason": "certificate missing"}


def _failed_receipt(cert_type: str, reason: str, *, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "type": cert_type,
        "present": True,
        "validator_receipt": False,
        "structural_validator_receipt": False,
        "no_data_use_ok": False,
        "reason": reason,
        "computed_outputs": {},
        "rows": rows or [],
    }


def _source_status(source_path: Path | None) -> dict[str, Any]:
    if source_path is None:
        return {"path": None, "present": False, "sha256": None}
    path = Path(source_path)
    return {
        "path": str(path),
        "present": path.exists(),
        "byte_size": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _template_no_data_use() -> dict[str, Any]:
    return {
        "observational_likelihoods_used": False,
        "used_data": [],
        "forbidden_data": list(FORBIDDEN_DATA_USE_KEYS),
    }


def _validation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cert_type, receipt in report.get("certificate_validations", {}).items():
        rows.append(
            {
                "type": cert_type,
                "present": receipt.get("present"),
                "validator_receipt": receipt.get("validator_receipt"),
                "structural_validator_receipt": receipt.get("structural_validator_receipt"),
                "no_data_use_ok": receipt.get("no_data_use_ok"),
                "reason": receipt.get("reason"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return None
    return array


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _float_or(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if _float_or_none(value) is not None]
    return float(np.mean(numeric)) if numeric else None


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["certificate_summary"]
    gates = report["readiness_gates"]
    lines = [
        "# OPH Inflation Certificate Bundle",
        "",
        f"- mode: `{report['mode']}`",
        f"- certificate directory: `{report.get('cert_dir')}`",
        f"- found/passed/expected: `{summary['found_count']}` / `{summary['passed_count']}` / `{summary['expected_count']}`",
        f"- inflation certificate stack ready: `{str(report['inflation_certificate_stack_ready']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: `{str(value).lower()}`" for name, value in gates.items())
    lines.extend(["", "## Certificates", ""])
    for cert_type, receipt in report["certificate_validations"].items():
        lines.append(
            f"- {cert_type}: present=`{str(receipt.get('present')).lower()}`, "
            f"receipt=`{str(receipt.get('validator_receipt')).lower()}`, "
            f"reason=`{receipt.get('reason') or 'ok'}`"
        )
    lines.extend(
        [
            "",
            "## Missing Types",
            "",
            ", ".join(summary["missing_types"]) or "none",
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
