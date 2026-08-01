"""Fail-closed issue-657 physical bi-refinement producer preflight.

The preflight consumes only the pinned CR-0 capability matrix.  It maps each
physical bi-refinement acceptance requirement to concrete CR-0 capability
rows, admits only simulator-native producers, and stops before an estimator or
comparison can run.  The issue-656 theorem packet is recorded as an external
mathematical prerequisite.  No sibling-repository bytes are claimed to have
been checked here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from oph_fpe.common_reserve.verify_capability_independent import (
    verify_capability_matrix,
)


SCHEMA = "oph.refinement.physical-birefinement-preflight.v1"
ISSUE = 657
CR0_ISSUE = 660
STATUS = "SOURCE_PRODUCER_MISSING"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CR0_MATRIX_PATH = (
    REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.json"
)
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/birefinement_preflight.schema.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/refinement/physical_birefinement_preflight.json"
DEFAULT_REPORT = REPOSITORY_ROOT / "data/refinement/physical_birefinement_preflight.md"

NATIVE = "AVAILABLE_SIMULATOR_NATIVE"

# These seven rows are a lossless decomposition of the bounded-work paragraph
# in issue 657.  Each row is tied to the CR-0 objects needed to instantiate it.
REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
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
    resolved = path.resolve(strict=True)
    resolved.relative_to(REPOSITORY_ROOT.resolve())
    raw = resolved.read_bytes()
    return {
        "path": resolved.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _load_cr0_matrix() -> dict[str, Any]:
    value = json.loads(CR0_MATRIX_PATH.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CR-0 capability matrix must be a JSON object")
    verification = verify_capability_matrix(value)
    if verification.get("receipt") is not True:
        reasons = verification.get("reasons") or ["independent CR-0 replay failed"]
        raise ValueError("; ".join(str(reason) for reason in reasons))
    return value


def _requirement_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    capabilities = matrix.get("capabilities")
    if not isinstance(capabilities, list):
        raise ValueError("CR-0 capability rows are missing")
    by_id = {
        row.get("capability_id"): row
        for row in capabilities
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for requirement_id, description, capability_ids in REQUIREMENTS:
        mapped = []
        for capability_id in capability_ids:
            capability = by_id.get(capability_id)
            if not isinstance(capability, Mapping):
                raise ValueError(f"CR-0 capability is missing: {capability_id}")
            classification = capability.get("classification")
            admitted = bool(
                classification == NATIVE
                and capability.get("machine_checks_passed") is True
                and capability.get("adapter_promotion_allowed") is False
            )
            mapped.append(
                {
                    "capability_id": capability_id,
                    "classification": classification,
                    "native_producer_admitted": admitted,
                    "theory_binding_status": capability.get("theory_binding_status"),
                }
            )
        blockers = [
            row["capability_id"]
            for row in mapped
            if row["native_producer_admitted"] is not True
        ]
        rows.append(
            {
                "requirement_id": requirement_id,
                "description": description,
                "capability_rows": mapped,
                "blocking_capability_ids": blockers,
                "requirement_satisfied": not blockers,
            }
        )
    return rows


def build_birefinement_preflight() -> dict[str, Any]:
    matrix = _load_cr0_matrix()
    requirements = _requirement_rows(matrix)
    if any(row["requirement_satisfied"] for row in requirements):
        raise ValueError("bounded missing-producer exit drifted")

    capability_rows = matrix["capabilities"]
    classifications = {
        classification: sorted(
            row["capability_id"]
            for row in capability_rows
            if row["classification"] == classification
        )
        for classification in matrix["allowed_classifications"]
    }
    blocking_ids = sorted(
        {
            capability_id
            for row in requirements
            for capability_id in row["blocking_capability_ids"]
        }
    )
    cr0_verification = verify_capability_matrix(matrix)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "status": STATUS,
        "scientific_promotion_allowed": False,
        "campaign_execution_allowed": False,
        "sources": {"cr0_capability_matrix": _raw_pin(CR0_MATRIX_PATH)},
        "cr0_replay": {
            "issue": CR0_ISSUE,
            "stage": "CR-0",
            "independent_verification_receipt": cr0_verification["receipt"],
            "independent_verification_reasons": cr0_verification["reasons"],
            "matrix_payload_sha256": matrix["payload_sha256"],
            "classification_counts": copy.deepcopy(matrix["classification_counts"]),
            "target_ancestry_audit_passed_for_cr0_roots": matrix["target_ancestry"][
                "passed"
            ],
            "candidate_producer_ancestry_audited": False,
        },
        "external_mathematical_prerequisite": {
            "issue": 656,
            "repository": "FloatingPragma/observer-patch-holography",
            "reported_status": (
                "THEOREM_PACKET_AND_MESH_SCAFFOLD_ATTAINED__"
                "SOURCE_NATIVE_PHYSICAL_BIREFINEMENT_OPEN"
            ),
            "scope": "external mathematical prerequisite only",
            "local_source_projection_vendored": False,
            "sibling_repository_bytes_verified": False,
            "physical_source_producer_supplied": False,
        },
        "admission_policy": {
            "admitted_classifications": [NATIVE],
            "conditional_objects_admitted": False,
            "control_objects_admitted": False,
            "adapter_promotion_admitted": False,
            "comparison_access": "REFUSED",
            "network_accessed": False,
            "environment_inputs_accessed": False,
            "public_data_accessed": False,
            "measurement_data_accessed": False,
            "target_values_accessed": False,
            "producer_input_paths": [
                "data/common_reserve/producer_capability_matrix.json"
            ],
        },
        "capability_classifications": classifications,
        "acceptance_requirements": requirements,
        "blocking_capability_ids": blocking_ids,
        "scientific_outputs": {
            "produced": False,
            "numeric_value_count": 0,
            "covariance_eigenvalues_emitted": False,
            "scaling_exponents_emitted": False,
            "comparison_statistics_emitted": False,
        },
        "decision": {
            "verdict": STATUS,
            "reason": (
                "The audited simulator catalog has no native physical quotient, "
                "support geometry, selected scalar register, OPH-native source ensemble, "
                "or common physical bi-refinement tower."
            ),
            "permitted_next_action": (
                "Implement and audit a missing source producer before constructing an "
                "eigenvalue estimator or opening comparison data."
            ),
        },
        "claim_boundary": (
            "This receipt establishes the bounded issue-657 SOURCE_PRODUCER_MISSING "
            "exit against the independently replayed CR-0 matrix. It supplies no "
            "physical refinement, scalar covariance, eigenvalue, scaling exponent, "
            "primordial prediction, or comparison."
        ),
    }
    result["payload_sha256"] = _sha(result)
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    Draft202012Validator(schema).validate(result)
    return result


def render_report(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# Physical bi-refinement producer preflight",
        "",
        f"Verdict: `{receipt['status']}`.",
        "",
        "The CR-0 capability matrix verifies independently. Native producer admission is required for every physical acceptance row.",
        "",
        "| Acceptance requirement | Blocking CR-0 capabilities |",
        "|---|---|",
    ]
    for row in receipt["acceptance_requirements"]:
        blockers = ", ".join(row["blocking_capability_ids"])
        lines.append(f"| {row['description']} | `{blockers}` |")
    lines.extend(
        [
            "",
            "The issue-656 theorem and mesh packet is recorded as an external mathematical prerequisite. No sibling-repository bytes are verified by this receipt.",
            "",
            "No covariance eigenvalues, scaling exponents, or comparison statistics are produced. Controls, adapters, measurement inputs, public data, and network access are refused.",
            "",
        ]
    )
    return "\n".join(lines)


def write_birefinement_preflight(
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    receipt = build_birefinement_preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", "utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(receipt), "utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    receipt = write_birefinement_preflight(args.out, args.report)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
