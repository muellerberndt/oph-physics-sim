from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.constants.oph_pixel import (
    P_SOURCE_MAP,
    P_STAR,
    PixelParameterProfile,
)
from oph_fpe.cosmology.edge_center_clock import (
    EDGE_CENTER_CLOCK_RECEIPT,
    EDGE_CENTER_EVIDENCE_RECEIPTS,
    FULL_COLLAR_DERIVATIVE_RECEIPT,
    GENERATIVE_PIXEL_PROFILE_RECEIPT,
    ORIENTATION_HALF_IDENTITY_RECEIPT,
    PHYSICAL_CLOCK_BINDING_RECEIPT,
    REFINEMENT_DEFECT_RECEIPT,
    SEMIGROUP_DEFECT_RECEIPT,
    SOURCE_DAG_CLEAN_RECEIPT,
    canonical_edge_clock_hash,
    edge_center_clock_target,
    validate_edge_center_clock_evidence,
    write_edge_center_clock_certificate,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/edge_center_clock_receipt.schema.json"
)


def _complete_evidence(
    *,
    P: float = P_SOURCE_MAP,
    profile: PixelParameterProfile = PixelParameterProfile.SOURCE_MAP,
) -> dict[str, object]:
    target = edge_center_clock_target(P)
    binding = {
        "clock_binding_source": "finite_refinement_clock_bundle",
        "full_collar_derivative": target.P / 24.0,
        "orientation_halves": 2,
        "orientation_half_identity_defect": 0.0,
        "semigroup_defect": 0.0,
        "refinement_defect": 0.0,
    }
    dag = {
        "nodes": [
            {
                "id": "full-collar-source",
                "kind": "source_theorem",
                "sha256": "sha256:" + "c" * 64,
            }
        ],
        "edges": [],
    }
    return {
        FULL_COLLAR_DERIVATIVE_RECEIPT: True,
        ORIENTATION_HALF_IDENTITY_RECEIPT: True,
        SEMIGROUP_DEFECT_RECEIPT: True,
        REFINEMENT_DEFECT_RECEIPT: True,
        PHYSICAL_CLOCK_BINDING_RECEIPT: True,
        SOURCE_DAG_CLEAN_RECEIPT: True,
        GENERATIVE_PIXEL_PROFILE_RECEIPT: True,
        "pixel_profile": profile.value,
        "clock_binding_payload": binding,
        "clock_binding_sha256": canonical_edge_clock_hash(binding),
        "source_dag": dag,
        "source_dag_sha256": canonical_edge_clock_hash(dag),
    }


def test_selected_edge_center_target_is_p_over_48_and_e_is_diagnostic() -> None:
    target = edge_center_clock_target()
    serialized = target.as_jsonable()

    assert target.full_collar_derivative == target.P / 24.0
    assert target.theta == target.P / 48.0
    assert target.n_s == 1.0 - target.P / 48.0
    assert target.kappa_rep == (target.P / 48.0) / (target.P - target.phi)
    assert abs(target.kappa_rep - 2.627023712627471) < 1.0e-12
    assert serialized["diagnostic_controls"]["e"]["kappa_rep"] == math.e
    assert serialized["diagnostic_controls"]["e"]["required"] is False
    assert serialized["diagnostic_controls"]["e"]["canonical"] is False
    assert serialized["diagnostic_controls"]["e"]["promoting"] is False


def test_edge_center_evidence_is_fail_closed_when_absent() -> None:
    evidence = validate_edge_center_clock_evidence()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert evidence[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert evidence["edge_center_clock_evidence_complete"] is False
    assert evidence["finite_step_survival_exponent_is_distinct"] is True
    assert tuple(evidence["missing_receipts"]) == EDGE_CENTER_EVIDENCE_RECEIPTS
    assert all(value is False for value in evidence["receipts"].values())
    Draft202012Validator(schema).validate(evidence)


def test_edge_center_evidence_requires_values_and_hash_bound_provenance() -> None:
    evidence = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )

    assert evidence[EDGE_CENTER_CLOCK_RECEIPT] is True
    assert evidence["missing_receipts"] == []


def test_copied_receipts_or_tampered_binding_hash_cannot_promote() -> None:
    packet = _complete_evidence()
    binding = dict(packet["clock_binding_payload"])
    binding["semigroup_defect"] = 1.0e-2
    packet["clock_binding_payload"] = binding

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert report["observed"]["clock_binding_hash_matches"] is False


def test_measurement_ancestry_cannot_promote_even_with_matching_hash() -> None:
    packet = _complete_evidence()
    dag = {
        "nodes": [
            {
                "id": "target",
                "kind": "measurement",
                "sha256": "sha256:" + "d" * 64,
            }
        ],
        "edges": [],
    }
    packet["source_dag"] = dag
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert report["observed"]["source_dag_audit"]["clean"] is False


def test_placeholder_source_node_digest_cannot_promote() -> None:
    packet = _complete_evidence()
    dag = {
        "nodes": [
            {
                "id": "full-collar-source",
                "kind": "source_theorem",
                "sha256": "sha256:" + "0" * 64,
            }
        ],
        "edges": [],
    }
    packet["source_dag"] = dag
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert "source_dag_node_hash_invalid" in report["observed"]["source_dag_audit"][
        "blockers"
    ]


def test_measured_endpoint_profile_cannot_promote_clock_bundle() -> None:
    packet = _complete_evidence(
        P=P_STAR,
        profile=PixelParameterProfile.MEASURED_COMPARISON,
    )

    report = validate_edge_center_clock_evidence(packet, P=P_STAR)

    assert report[GENERATIVE_PIXEL_PROFILE_RECEIPT] is False
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert "pixel_profile_is_comparison_only" in report["observed"]["pixel_provenance"][
        "blockers"
    ]


def test_standalone_certificate_writer_validates_strict_schema(tmp_path: Path) -> None:
    report = write_edge_center_clock_certificate(
        tmp_path,
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )
    persisted = json.loads(
        (tmp_path / "edge_center_clock_certificate.json").read_text(encoding="utf-8")
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert persisted == report
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
