"""Independent verifier for the issue-657 missing-producer receipt.

This module does not import the issue-657 producer.  It reloads the sole CR-0
input, runs the independent CR-0 verifier, reconstructs every requirement map,
checks the comparison firewall, and rejects any scientific output promotion.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from oph_fpe.common_reserve.verify_capability_independent import (
    verify_capability_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = REPOSITORY_ROOT / "data/refinement/physical_birefinement_preflight.json"
CR0_PATH = REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.json"
SCHEMA_PATH = REPOSITORY_ROOT / "oph_fpe/refinement/schemas/birefinement_preflight.schema.json"
PRODUCER_PATH = REPOSITORY_ROOT / "oph_fpe/refinement/birefinement_preflight.py"

NATIVE = "AVAILABLE_SIMULATOR_NATIVE"
EXPECTED_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "carrier_support_metric",
        "Carrier and support objects with a carrier-to-support metric",
        ("raw_twelve_port_response", "physical_quotient", "support_geometry"),
    ),
    (
        "quotient_observable_birefinement",
        "Exact commuting R2, R3, and R6 maps on quotient states and observable algebras",
        ("physical_quotient", "refinement_tower_physical_scale_ratios"),
    ),
    (
        "common_scalar_covariance_family",
        "One quotient-visible scalar with projector, source ensemble, precision operator, and route covariances",
        (
            "physical_quotient",
            "scalar_register",
            "source_ensemble_action",
            "refinement_tower_physical_scale_ratios",
        ),
    ),
    (
        "physical_linear_resolution_ratios",
        "Source-emitted linear resolutions with exact two-fold and three-fold ratios, excluding count and area proxies",
        ("support_geometry", "refinement_tower_physical_scale_ratios"),
    ),
    (
        "isolated_common_covariance_eigenray",
        "One common isolated covariance eigenray with projector, interval, leakage, gap, and route certificates",
        (
            "scalar_register",
            "source_ensemble_action",
            "refinement_tower_physical_scale_ratios",
        ),
    ),
    (
        "cofinal_summable_convergence",
        "Cofinal convergence on one lineage with a summable certified error budget",
        (
            "scalar_register",
            "source_ensemble_action",
            "refinement_tower_physical_scale_ratios",
        ),
    ),
    (
        "target_free_candidate_provenance",
        "A candidate-producer provenance DAG with no target or calibrated-proxy ancestry",
        (
            "physical_quotient",
            "scalar_register",
            "source_ensemble_action",
            "refinement_tower_physical_scale_ratios",
        ),
    ),
)

EXPECTED_DECISION = {
    "verdict": "SOURCE_PRODUCER_MISSING",
    "reason": (
        "The audited simulator catalog has no native physical quotient, support "
        "geometry, selected scalar register, OPH-native source ensemble, or common "
        "physical bi-refinement tower."
    ),
    "permitted_next_action": (
        "Implement and audit a missing source producer before constructing an "
        "eigenvalue estimator or opening comparison data."
    ),
}
EXPECTED_CLAIM_BOUNDARY = (
    "This receipt establishes the bounded issue-657 SOURCE_PRODUCER_MISSING exit "
    "against the independently replayed CR-0 matrix. It supplies no physical "
    "refinement, scalar covariance, eigenvalue, scaling exponent, primordial "
    "prediction, or comparison."
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _expected_requirements(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("capabilities")
    if not isinstance(rows, list):
        raise ValueError("CR-0 capability rows are absent")
    by_id = {
        row.get("capability_id"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = []
    for requirement_id, description, capability_ids in EXPECTED_REQUIREMENTS:
        mapped = []
        for capability_id in capability_ids:
            capability = by_id.get(capability_id)
            if not isinstance(capability, Mapping):
                raise ValueError(f"CR-0 capability is absent: {capability_id}")
            admitted = bool(
                capability.get("classification") == NATIVE
                and capability.get("machine_checks_passed") is True
                and capability.get("adapter_promotion_allowed") is False
            )
            mapped.append(
                {
                    "capability_id": capability_id,
                    "classification": capability.get("classification"),
                    "native_producer_admitted": admitted,
                    "theory_binding_status": capability.get("theory_binding_status"),
                }
            )
        blockers = [
            row["capability_id"]
            for row in mapped
            if row["native_producer_admitted"] is not True
        ]
        expected.append(
            {
                "requirement_id": requirement_id,
                "description": description,
                "capability_rows": mapped,
                "blocking_capability_ids": blockers,
                "requirement_satisfied": not blockers,
            }
        )
    return expected


def _static_io_reasons() -> list[str]:
    reasons: list[str] = []
    source = PRODUCER_PATH.read_text("utf-8")
    tree = ast.parse(source, filename=str(PRODUCER_PATH))
    forbidden_roots = {
        "aiohttp",
        "boto3",
        "httpx",
        "os",
        "pandas",
        "requests",
        "socket",
        "subprocess",
        "urllib",
    }
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        for name in names:
            if name.split(".", 1)[0] in forbidden_roots:
                reasons.append(f"forbidden_producer_import:{name}")
    for forbidden_arg in ("--matrix", "--source", "--input", "--comparison"):
        if forbidden_arg in source:
            reasons.append(f"producer_accepts_external_input:{forbidden_arg}")
    for forbidden_path in (
        "data/measurements/",
        "data/public/",
        "public_data/",
        "comparison_data/",
    ):
        if forbidden_path in source:
            reasons.append(f"producer_mentions_comparison_path:{forbidden_path}")
    return reasons


def verify_birefinement_preflight(receipt: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        schema = _load_json(SCHEMA_PATH)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(receipt),
            key=lambda error: list(error.path),
        )
        reasons.extend(
            f"schema:{'/'.join(map(str, error.path))}:{error.message}"
            for error in errors
        )

        payload = copy.deepcopy(dict(receipt))
        digest = payload.pop("payload_sha256", None)
        if digest != _sha(payload):
            reasons.append("payload_sha256_mismatch")

        matrix = _load_json(CR0_PATH)
        cr0_verification = verify_capability_matrix(matrix)
        if cr0_verification.get("receipt") is not True:
            reasons.append("cr0_independent_replay_failed")
            reasons.extend(
                f"cr0:{reason}" for reason in cr0_verification.get("reasons", [])
            )
        sources = receipt.get("sources")
        expected_sources = {"cr0_capability_matrix": _raw_pin(CR0_PATH)}
        if sources != expected_sources:
            reasons.append("cr0_source_pin_mismatch")

        expected_cr0_replay = {
            "issue": 660,
            "stage": "CR-0",
            "independent_verification_receipt": True,
            "independent_verification_reasons": [],
            "matrix_payload_sha256": matrix.get("payload_sha256"),
            "classification_counts": matrix.get("classification_counts"),
            "target_ancestry_audit_passed_for_cr0_roots": (
                (matrix.get("target_ancestry") or {}).get("passed")
            ),
            "candidate_producer_ancestry_audited": False,
        }
        if receipt.get("cr0_replay") != expected_cr0_replay:
            reasons.append("cr0_replay_summary_mismatch")

        capabilities = matrix.get("capabilities")
        if not isinstance(capabilities, list):
            raise ValueError("CR-0 capabilities are malformed")
        expected_classifications = {
            classification: sorted(
                row["capability_id"]
                for row in capabilities
                if row["classification"] == classification
            )
            for classification in matrix["allowed_classifications"]
        }
        if receipt.get("capability_classifications") != expected_classifications:
            reasons.append("capability_classification_projection_mismatch")

        expected_requirements = _expected_requirements(matrix)
        if receipt.get("acceptance_requirements") != expected_requirements:
            reasons.append("acceptance_requirement_mapping_mismatch")
        expected_blockers = sorted(
            {
                capability_id
                for row in expected_requirements
                for capability_id in row["blocking_capability_ids"]
            }
        )
        if receipt.get("blocking_capability_ids") != expected_blockers:
            reasons.append("blocking_capability_projection_mismatch")
        if any(row["requirement_satisfied"] for row in expected_requirements):
            reasons.append("bounded_exit_not_forced_by_matrix")

        if receipt.get("decision") != EXPECTED_DECISION:
            reasons.append("decision_mismatch")
        if receipt.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
            reasons.append("claim_boundary_mismatch")
        reasons.extend(_static_io_reasons())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        reasons.append(f"verification_exception:{type(exc).__name__}:{exc}")
    return {
        "schema": "oph.refinement.physical-birefinement-preflight-independent-verification.v1",
        "issue": 657,
        "receipt": not reasons,
        "verdict": "VERIFIED_SOURCE_PRODUCER_MISSING" if not reasons else "REFUSED",
        "reasons": sorted(set(reasons)),
        "scientific_promotion_allowed": False,
        "comparison_access": "REFUSED",
    }


def verify_file(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    try:
        return verify_birefinement_preflight(_load_json(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema": "oph.refinement.physical-birefinement-preflight-independent-verification.v1",
            "issue": 657,
            "receipt": False,
            "verdict": "REFUSED",
            "reasons": [f"receipt_parse_failed:{type(exc).__name__}:{exc}"],
            "scientific_promotion_allowed": False,
            "comparison_access": "REFUSED",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    result = verify_file(args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
