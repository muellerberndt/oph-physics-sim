from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import pytest

from oph_fpe.constants.oph_pixel import (
    P_SOURCE_MAP,
    P_STAR,
    PixelParameterProfile,
)
from oph_fpe.cosmology.edge_center_clock import (
    CLOCK_BINDING_PACKET_CONSISTENCY_RECEIPT,
    EDGE_CENTER_CLOCK_RECEIPT,
    EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT,
    EDGE_CENTER_EVIDENCE_RECEIPTS,
    FULL_COLLAR_DERIVATIVE_RECEIPT,
    GENERATIVE_PIXEL_PROFILE_RECEIPT,
    INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT,
    MAX_DEFECT_TOLERANCE,
    MAX_SOURCE_DAG_NODES,
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
    binding_sha256 = canonical_edge_clock_hash(binding)
    dag = {
        "nodes": [
            {
                "id": "full-collar-source",
                "kind": "source_theorem",
                "sha256": "sha256:" + "c" * 64,
            },
            {
                "id": "finite_refinement_clock_bundle",
                "kind": "clock_binding",
                "sha256": binding_sha256,
            },
        ],
        "edges": [
            {
                "from": "full-collar-source",
                "to": "finite_refinement_clock_bundle",
            }
        ],
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
        "clock_binding_sha256": binding_sha256,
        "source_dag": dag,
        "source_dag_sha256": canonical_edge_clock_hash(dag),
    }


def _rehash_binding(packet: dict[str, object]) -> None:
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding_sha256 = canonical_edge_clock_hash(binding)
    packet["clock_binding_sha256"] = binding_sha256
    dag = packet["source_dag"]
    assert isinstance(dag, dict)
    nodes = dag["nodes"]
    assert isinstance(nodes, list)
    source_id = binding.get("clock_binding_source")
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == source_id:
            node["sha256"] = binding_sha256
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)


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


def test_edge_center_packet_consistency_does_not_promote_open_finite_run_gate() -> None:
    evidence = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )

    assert evidence[EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT] is True
    assert evidence[CLOCK_BINDING_PACKET_CONSISTENCY_RECEIPT] is True
    assert evidence[PHYSICAL_CLOCK_BINDING_RECEIPT] is False
    assert evidence[INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT] is False
    assert evidence[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert evidence["missing_receipts"] == [
        PHYSICAL_CLOCK_BINDING_RECEIPT,
        INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT,
    ]
    assert evidence["observed"]["source_dag_audit"]["binding_node_is_leaf"] is True
    assert (
        evidence["observed"]["source_dag_audit"]["root_source_reaches_binding"]
        is True
    )


def test_caller_receipt_booleans_are_ignored() -> None:
    packet = _complete_evidence()
    for receipt_name in EDGE_CENTER_EVIDENCE_RECEIPTS:
        packet[receipt_name] = False

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT] is True
    assert report["legacy_receipt_declarations_promoted"] is False
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


@pytest.mark.parametrize(
    "tolerance",
    [True, False, -1.0, math.inf, -math.inf, math.nan, 1.0e6, 10**400],
)
def test_tolerance_must_use_the_frozen_bounded_numeric_profile(
    tolerance: object,
) -> None:
    with pytest.raises(ValueError):
        validate_edge_center_clock_evidence(
            _complete_evidence(),
            P=P_SOURCE_MAP,
            tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_tolerance_cannot_exceed_frozen_maximum() -> None:
    with pytest.raises(ValueError):
        validate_edge_center_clock_evidence(
            _complete_evidence(),
            P=P_SOURCE_MAP,
            tolerance=MAX_DEFECT_TOLERANCE * 2.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_collar_derivative", True),
        ("orientation_half_identity_defect", False),
        ("semigroup_defect", False),
        ("refinement_defect", False),
    ],
)
def test_boolean_numeric_evidence_cannot_promote(field: str, value: bool) -> None:
    packet = _complete_evidence()
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding[field] = value
    _rehash_binding(packet)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_nonfinite_defect_is_rejected_before_hashing(value: float) -> None:
    packet = _complete_evidence()
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding["semigroup_defect"] = value

    with pytest.raises(ValueError):
        canonical_edge_clock_hash(binding)


def test_overflowing_binding_scalar_fails_packet_consistency_without_crashing() -> None:
    packet = _complete_evidence()
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding["full_collar_derivative"] = 10**400
    _rehash_binding(packet)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT] is False
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


def test_finite_defect_above_frozen_bound_cannot_promote() -> None:
    packet = _complete_evidence()
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding["semigroup_defect"] = MAX_DEFECT_TOLERANCE * 2.0
    _rehash_binding(packet)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[SEMIGROUP_DEFECT_RECEIPT] is False
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


def test_clock_binding_source_must_name_an_actual_dag_node() -> None:
    packet = _complete_evidence()
    binding = packet["clock_binding_payload"]
    assert isinstance(binding, dict)
    binding["clock_binding_source"] = "missing-clock-binding"
    _rehash_binding(packet)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[PHYSICAL_CLOCK_BINDING_RECEIPT] is False
    assert "clock_binding_source_node_missing" in report["observed"][
        "source_dag_audit"
    ]["blockers"]


def test_clock_binding_source_node_digest_must_equal_binding_digest() -> None:
    packet = _complete_evidence()
    dag = packet["source_dag"]
    assert isinstance(dag, dict)
    nodes = dag["nodes"]
    assert isinstance(nodes, list)
    binding_node = next(
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("id") == "finite_refinement_clock_bundle"
    )
    binding_node["sha256"] = "sha256:" + "d" * 64
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert report[PHYSICAL_CLOCK_BINDING_RECEIPT] is False
    assert "clock_binding_source_hash_mismatch" in report["observed"][
        "source_dag_audit"
    ]["blockers"]


def test_clock_binding_source_must_be_a_leaf_in_an_acyclic_dag() -> None:
    packet = _complete_evidence()
    dag = packet["source_dag"]
    assert isinstance(dag, dict)
    edges = dag["edges"]
    assert isinstance(edges, list)
    edges.append(
        {
            "from": "finite_refinement_clock_bundle",
            "to": "full-collar-source",
        }
    )
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    blockers = report["observed"]["source_dag_audit"]["blockers"]
    assert "source_dag_cycle" in blockers
    assert "clock_binding_source_not_leaf" in blockers
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


def test_source_dag_node_limit_is_fail_closed() -> None:
    packet = _complete_evidence()
    dag = packet["source_dag"]
    assert isinstance(dag, dict)
    dag["nodes"] = [
        {
            "id": f"source-{index}",
            "kind": "source_theorem",
            "sha256": "sha256:" + f"{index + 1:064x}",
        }
        for index in range(MAX_SOURCE_DAG_NODES + 1)
    ]
    dag["edges"] = []
    packet["source_dag_sha256"] = canonical_edge_clock_hash(dag)

    report = validate_edge_center_clock_evidence(packet, P=P_SOURCE_MAP)

    assert "source_dag_node_limit_exceeded" in report["observed"][
        "source_dag_audit"
    ]["blockers"]
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False


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
    replay = json.loads(
        (tmp_path / "edge_center_clock_input_packet.json").read_text(encoding="utf-8")
    )
    assert replay["evidence"] == _complete_evidence()
    assert replay["P"] == P_SOURCE_MAP
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_schema_rejects_forged_positive_physical_clock_receipt() -> None:
    report = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered[EDGE_CENTER_CLOCK_RECEIPT] = True
    tampered["edge_center_clock_evidence_complete"] = True

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(tampered)


def test_schema_rejects_forged_independent_finite_run_receipt() -> None:
    report = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered[INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT] = True
    tampered["receipts"][INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT] = True

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(tampered)


def test_schema_rejects_forged_physical_clock_binding_subreceipt() -> None:
    report = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered[PHYSICAL_CLOCK_BINDING_RECEIPT] = True
    tampered["receipts"][PHYSICAL_CLOCK_BINDING_RECEIPT] = True

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(tampered)


def test_schema_binds_packet_consistency_to_nested_checks() -> None:
    report = validate_edge_center_clock_evidence(
        _complete_evidence(),
        P=P_SOURCE_MAP,
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered["checks"]["semigroup_defect_bounded"] = False

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(tampered)
