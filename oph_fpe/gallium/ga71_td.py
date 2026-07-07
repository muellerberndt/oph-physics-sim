from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATIOS = REPO_ROOT / "data" / "gallium" / "public" / "ground_state_ratios.json"
DEFAULT_WITNESS = REPO_ROOT / "data" / "gallium" / "benchmarks" / "mdg_witness.json"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "gallium" / "evidence" / "ga71_td" / "manifest.template.json"

PROTECTED_OUTPUTS = ("rho_TD_71", "S_TD_71", "sigma_gs_51Cr", "sigma_gs_37Ar")
FORBIDDEN_FLAGS = {"FORBIDDEN_TARGET"}
VALID_MEASUREMENT_FLAGS = {"NONE", "FORBIDDEN_TARGET", "COMPARISON_ONLY"}
STATUS_ENUM = (
    "TEMPLATE_NOT_VALID_FOR_PROMOTION",
    "GA71_TD_CERTIFICATE_AWAITING_SOURCE_SIMULATION",
    "PASSED_SOURCE_SIDE",
    "PASSED_WITH_EXTERNAL_FIXED_INPUTS",
    "FAILED",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def compute_mdg_forward_comparison(
    ratios: dict[str, Any] | None = None,
    witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratios = ratios or load_json(DEFAULT_RATIOS)
    witness = witness or load_json(DEFAULT_WITNESS)
    reference = witness["detailed_balance_reference"]
    mdg = witness["mdg_witness"]
    kappa = {
        "51Cr": mdg["sigma_gs_51Cr_1e-45_cm2"] / reference["sigma_gs_51Cr_1e-45_cm2"],
        "37Ar": mdg["sigma_gs_37Ar_1e-45_cm2"] / reference["sigma_gs_37Ar_1e-45_cm2"],
    }
    corrected = []
    for row in ratios["rows"]:
        source = row["source"]
        if source not in kappa:
            raise ValueError(f"unsupported source {source!r}")
        factor = kappa[source]
        value = float(row["R_gs"]) / factor
        sigma = float(row["sigma_R_gs"]) / factor
        corrected.append(
            {
                "run_id": row["run_id"],
                "experiment": row["experiment"],
                "source": source,
                "R_gs": row["R_gs"],
                "sigma_R_gs": row["sigma_R_gs"],
                "kappa": factor,
                "corrected_R": value,
                "corrected_sigma": sigma,
                "measurement_use_flag": "COMPARISON_ONLY",
            }
        )
    weights = np.array([1.0 / (row["corrected_sigma"] ** 2) for row in corrected], dtype=float)
    values = np.array([row["corrected_R"] for row in corrected], dtype=float)
    weighted_mean = float(np.sum(weights * values) / np.sum(weights))
    weighted_sigma = float(1.0 / np.sqrt(np.sum(weights)))
    return {
        "schema": "oph_ga71_mdg_forward_comparison_v1",
        "status": "COMPARISON_ONLY_NOT_SOURCE_CERTIFICATE",
        "kappa_51Cr": kappa["51Cr"],
        "kappa_37Ar": kappa["37Ar"],
        "rows": corrected,
        "all_source_weighted_mean": {
            "value": weighted_mean,
            "sigma": weighted_sigma,
            "covariance_model": "diagonal_diagnostic",
        },
        "nonclaims": [
            "This forward comparison is diagnostic only.",
            "It must be generated after source artifacts are frozen in any promotion-valid run.",
            "It cannot be used to tune rho_TD_71 or choose a nuclear selector.",
        ],
    }


def build_no_target_leak_dag(manifest: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: dict(node) for node in manifest.get("nodes", [])}
    taint_cache: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}

    def tainted(node_id: str, stack: tuple[str, ...] = ()) -> bool:
        if node_id in taint_cache:
            return taint_cache[node_id]
        if node_id not in nodes:
            taint_cache[node_id] = True
            reasons[node_id] = ["missing_node_metadata_fails_closed"]
            return True
        if node_id in stack:
            taint_cache[node_id] = True
            reasons[node_id] = ["cycle_detected_fails_closed"]
            return True
        node = nodes[node_id]
        flag = node.get("measurement_use_flag")
        node_reasons: list[str] = []
        if flag not in VALID_MEASUREMENT_FLAGS:
            node_reasons.append("unknown_measurement_use_flag_fails_closed")
        if flag in FORBIDDEN_FLAGS:
            node_reasons.append("forbidden_target_node")
        parent_taints = []
        for parent in node.get("parents", []) or []:
            if tainted(parent, stack + (node_id,)):
                parent_taints.append(parent)
        if parent_taints:
            node_reasons.append("tainted_parents:" + ",".join(parent_taints))
        taint_cache[node_id] = bool(node_reasons)
        reasons[node_id] = node_reasons
        return taint_cache[node_id]

    protected = tuple(manifest.get("protected_outputs", PROTECTED_OUTPUTS))
    dag_nodes = []
    for node_id, node in nodes.items():
        node_tainted = tainted(node_id)
        dag_nodes.append(
            {
                "id": node_id,
                "parents": list(node.get("parents", []) or []),
                "type": node.get("type"),
                "measurement_use_flag": node.get("measurement_use_flag"),
                "forbidden_target_ancestor": node_tainted,
                "taint_reasons": reasons.get(node_id, []),
                "protected_output": node_id in protected,
            }
        )
    protected_failures = [node_id for node_id in protected if tainted(node_id)]
    return {
        "schema": "oph_ga71_no_target_leak_dag_v1",
        "protected_outputs": list(protected),
        "nodes": sorted(dag_nodes, key=lambda row: row["id"]),
        "protected_failures": protected_failures,
        "no_target_leak_pass": not protected_failures,
        "claim_boundary": "A pass only certifies DAG taint structure for the provided manifest. It does not certify source-side nuclear physics.",
    }


def validate_no_target_leak(manifest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    dag = build_no_target_leak_dag(manifest)
    return bool(dag["no_target_leak_pass"]), dag


def build_template_certificate(
    *,
    manifest: dict[str, Any] | None = None,
    witness: dict[str, Any] | None = None,
    ratios: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or load_json(DEFAULT_MANIFEST)
    witness = witness or load_json(DEFAULT_WITNESS)
    ratios = ratios or load_json(DEFAULT_RATIOS)
    no_target_leak_pass, dag = validate_no_target_leak(manifest)
    forward = compute_mdg_forward_comparison(ratios=ratios, witness=witness)
    cert = {
        "schema_version": "oph_ga71_td_source_certificate_v1",
        "certificate_id": "GA71_TD_SOURCE_CERTIFICATE",
        "status": "TEMPLATE_NOT_VALID_FOR_PROMOTION",
        "claim_boundary": (
            "Template and audit scaffold only. A promotion-valid certificate must be emitted "
            "by a frozen source-side A=71 nuclear calculation, not by mDG witness copying."
        ),
        "target_gates": witness["target_bands"],
        "source_provenance_firewall": {
            "protected_outputs": list(manifest.get("protected_outputs", PROTECTED_OUTPUTS)),
            "forbidden_upstream_inputs": manifest.get("forbidden_upstream_inputs", []),
            "allowed_source_inputs": manifest.get("allowed_source_inputs", []),
            "no_target_leak_dag_status": "PASS_TEMPLATE_STRUCTURE" if no_target_leak_pass else "FAILED",
            "no_target_leak_pass": no_target_leak_pass,
        },
        "artifacts": {
            "source_manifest": "evidence/ga71_td/manifest.json",
            "no_target_leak_dag": "evidence/ga71_td/dag/no_target_leak.json",
            "rho_TD_71": None,
            "S_TD_71": None,
            "ec_result": None,
            "ibd_result": None,
            "uncertainty": None,
            "forward_comparison": "evidence/ga71_td/forward_compare/results.json",
        },
        "density": {
            "rho_grid_fm": None,
            "rho_values": None,
            "normalization": None,
            "sign_changes": [],
            "basis": None,
            "hamiltonian": None,
            "axial_current_operator": None,
            "obtd_source_hash": None,
            "uncertainty_ensemble": None,
            "status": "SOURCE_SIMULATION_REQUIRED",
        },
        "ec_half_life": {
            "T_half_71Ge_days": None,
            "target_days": witness["target_bands"]["T_half_71Ge_days"],
            "status": "PENDING_SOURCE_SIMULATION",
        },
        "ibd_cross_sections": {
            "sigma_gs_51Cr_1e-45_cm2": None,
            "sigma_gs_37Ar_1e-45_cm2": None,
            "target_sigma_gs_51Cr_1e-45_cm2": witness["target_bands"]["sigma_gs_51Cr_1e-45_cm2"],
            "target_sigma_gs_37Ar_1e-45_cm2": witness["target_bands"]["sigma_gs_37Ar_1e-45_cm2"],
            "status": "PENDING_SOURCE_SIMULATION",
        },
        "uncertainty_ledger": {
            "status": "PENDING",
            "required_terms": [
                "nuclear Hamiltonian / interaction variation",
                "basis truncation",
                "single-particle radial basis",
                "effective axial current / quenching convention",
                "spectroscopy constraint envelope",
                "charge/radius input uncertainty",
                "Q_EC uncertainty",
                "71Ge half-life uncertainty",
                "atomic Dirac solver tolerance",
                "radiative and weak-magnetism conventions",
                "source line energies and branching fractions",
                "quadrature/grid convergence",
                "model covariance",
            ],
        },
        "forward_comparison": forward,
        "gates": {
            "NO_TARGET_LEAK_DAG": "PASS_TEMPLATE_STRUCTURE" if no_target_leak_pass else "FAILED",
            "SOURCE_NUCLEAR_SELECTOR": "PENDING",
            "EC_HALFLIFE_REPRODUCED": "PENDING",
            "IBD_CROSS_SECTIONS_IN_RANGE": "PENDING",
            "UNCERTAINTY_LEDGER_PRESENT": "PENDING",
            "FORWARD_COMPARISON_PRESENT": "DIAGNOSTIC_PRESENT",
            "REPRODUCIBLE_HASHED_ARTIFACTS": "PENDING",
        },
        "no_target_leak_dag": dag,
        "hashes": {
            "manifest_template_sha256": _json_sha256(manifest),
            "mdg_witness_sha256": _json_sha256(witness),
            "ground_state_ratios_sha256": _json_sha256(ratios),
        },
        "nonclaims": [
            "This JSON is a template and is not valid for promotion.",
            "mDG/mTG benchmark values are witness/comparison rows, not source-side selector outputs.",
            "No gallium source anomaly closure is claimed until source-side density, EC, IBD, uncertainty, and replay hashes exist.",
            "Forward comparison may not be an upstream parent of rho_TD_71 or sigma_gs outputs.",
        ],
    }
    cert["hashes"]["certificate_payload_sha256"] = _json_sha256(_without_certificate_payload_hash(cert))
    return cert


def validate_certificate(cert: dict[str, Any]) -> dict[str, Any]:
    required = (
        "certificate_id",
        "status",
        "source_provenance_firewall",
        "density",
        "ec_half_life",
        "ibd_cross_sections",
        "uncertainty_ledger",
        "forward_comparison",
        "gates",
        "hashes",
        "nonclaims",
    )
    missing = [key for key in required if key not in cert]
    invalid_status = cert.get("status") not in STATUS_ENUM
    promotion_status = cert.get("status") in {"PASSED_SOURCE_SIDE", "PASSED_WITH_EXTERNAL_FIXED_INPUTS"}
    pending_gate_names = [
        key for key, value in cert.get("gates", {}).items() if str(value).startswith("PENDING")
    ]
    source_artifacts_present = bool(cert.get("density", {}).get("rho_grid_fm")) and bool(
        cert.get("ibd_cross_sections", {}).get("sigma_gs_51Cr_1e-45_cm2")
    )
    promotion_valid = (
        not missing
        and not invalid_status
        and promotion_status
        and not pending_gate_names
        and source_artifacts_present
        and cert.get("source_provenance_firewall", {}).get("no_target_leak_pass") is True
    )
    return {
        "schema": "oph_ga71_certificate_validation_v1",
        "valid_json_shape": not missing and not invalid_status,
        "promotion_valid": promotion_valid,
        "missing_required_fields": missing,
        "invalid_status": invalid_status,
        "pending_gates": pending_gate_names,
        "source_artifacts_present": source_artifacts_present,
        "status": "PROMOTION_VALID" if promotion_valid else "NOT_PROMOTION_VALID",
    }


def write_ga71_template_bundle(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(DEFAULT_MANIFEST)
    witness = load_json(DEFAULT_WITNESS)
    ratios = load_json(DEFAULT_RATIOS)
    cert = build_template_certificate(manifest=manifest, witness=witness, ratios=ratios)
    validation = validate_certificate(cert)
    dag = cert["no_target_leak_dag"]
    forward = cert["forward_comparison"]
    schema = _schema()

    schema_path = out_dir / "GA71_TD_SOURCE_CERTIFICATE.schema.json"
    template_path = out_dir / "GA71_TD_SOURCE_CERTIFICATE.template.json"
    validation_path = out_dir / "GA71_TD_SOURCE_CERTIFICATE.validation.json"
    dag_path = out_dir / "no_target_leak.template.json"
    forward_path = out_dir / "mdg_forward_compare.diagnostic.json"
    instructions_path = out_dir / "ga71_certificate_coding_agent_instructions.md"

    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    template_path.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    dag_path.write_text(json.dumps(dag, indent=2, sort_keys=True), encoding="utf-8")
    forward_path.write_text(json.dumps(forward, indent=2, sort_keys=True), encoding="utf-8")
    instructions_path.write_text(_instructions_markdown(), encoding="utf-8")

    return {
        "schema": "oph_ga71_template_bundle_report_v1",
        "status": cert["status"],
        "promotion_valid": validation["promotion_valid"],
        "template": str(template_path),
        "schema_path": str(schema_path),
        "validation": str(validation_path),
        "no_target_leak_dag": str(dag_path),
        "forward_comparison": str(forward_path),
        "instructions": str(instructions_path),
        "nonclaims": cert["nonclaims"],
    }


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://floatingpragma.io/schemas/ga71_td_source_certificate_v1.json",
        "title": "GA71 Transition Density Source Certificate",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "certificate_id",
            "status",
            "source_provenance_firewall",
            "density",
            "ec_half_life",
            "ibd_cross_sections",
            "uncertainty_ledger",
            "forward_comparison",
            "gates",
            "hashes",
            "nonclaims",
        ],
        "properties": {
            "certificate_id": {"type": "string"},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
            "source_provenance_firewall": {"type": "object"},
            "density": {"type": "object"},
            "ec_half_life": {"type": "object"},
            "ibd_cross_sections": {"type": "object"},
            "uncertainty_ledger": {"type": "object"},
            "forward_comparison": {"type": "object"},
            "gates": {"type": "object"},
            "hashes": {"type": "object"},
            "nonclaims": {"type": "array", "items": {"type": "string"}},
        },
    }


def _instructions_markdown() -> str:
    return """# GA71_TD_SOURCE_CERTIFICATE Coding Instructions

This bundle is a template only. Do not close or promote the gallium anomaly from it.

Promotion-valid statuses are `PASSED_SOURCE_SIDE` or `PASSED_WITH_EXTERNAL_FIXED_INPUTS`,
and they require real source-side artifacts:

- frozen source manifest and no-target-leak DAG;
- source-produced `rho_TD_71` transition density with sign-change evidence;
- EC half-life result near 11.465 d;
- IBD cross sections in the declared 51Cr/37Ar bands;
- full uncertainty ledger;
- forward comparison performed only after source artifacts are frozen.

Forbidden upstream inputs include GALLEX/SAGE/BEST residuals, inferred gallium source
cross sections, sterile-fit residuals, and posterior anomaly residuals. Unknown DAG
metadata fails closed.

The mDG witness file is a benchmark and diagnostic comparison only. Copying its values
into the certificate is not source-side evidence.
"""


def _without_certificate_payload_hash(cert: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(cert))
    clone.setdefault("hashes", {}).pop("certificate_payload_sha256", None)
    return clone


def _json_sha256(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
