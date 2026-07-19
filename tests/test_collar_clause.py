from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from oph_fpe.cosmology.collar_clause import (
    PINNED_COLLAR_THEORY_ARTIFACTS,
    canonical_evidence_hash,
    collar_clause_negative_controls,
    verify_collar_clause_packet,
    write_collar_clause_certificate,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/collar_clause_receipt.schema.json"
)


def _content_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _packet(
    *,
    cross_coordinates: list[float] | None = None,
    include_factorization: bool = True,
    diagnostics: dict[str, bool] | None = None,
    source_kind: str = "simulation_derivation",
) -> dict:
    ancestry = [
        {
            "node_id": node_id,
            "kind": "lean_theorem" if "clause" in node_id else "lean_definition",
            "sha256": digest,
        }
        for node_id, digest in PINNED_COLLAR_THEORY_ARTIFACTS.items()
    ] + [
        {
            "node_id": "run-collar-family-derivation",
            "kind": source_kind,
            "sha256": _content_hash("CollarLayer.collarClause_posFamily"),
        }
    ]
    layer = {
        "ambient_dimension": 4,
        "left_basis": [[0.0, 0.0, 1.0, 0.0]],
        "right_basis": [[0.0, 0.0, 0.0, 1.0]],
        "flux_basis": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    }
    cross = cross_coordinates or [2.0, -1.0, 0.0, 0.0]
    densities = [
        {
            "density_id": "left-density",
            "coordinates": [0.0, 0.0, 3.0, 0.0],
            "declared_support": "LEFT",
        },
        {
            "density_id": "right-density",
            "coordinates": [0.0, 0.0, 0.0, -2.0],
            "declared_support": "RIGHT",
        },
        {
            "density_id": "cross-cut-density",
            "coordinates": cross,
            "declared_support": "CROSS_CUT",
        },
    ]
    if include_factorization:
        witness_core = {
            "flux_basis_hash": canonical_evidence_hash(layer["flux_basis"]),
            "flux_coefficients": [2.0, -1.0],
            "source_derivation_ids": ["run-collar-family-derivation"],
            "refinement_natural": True,
        }
        densities[-1]["flux_factorization"] = {
            **witness_core,
            "witness_hash": canonical_evidence_hash(witness_core),
        }
    family_laws = {
        "manifest_complete": True,
        "gauge_invariant": True,
        "collar_supported": True,
        "refinement_closed": True,
        "refinement_channel_manifest_complete": True,
        "source_derivation_ids": ["run-collar-family-derivation"],
    }
    return {
        "packet_id": "bounded-collar-positive",
        "source_ancestry": ancestry,
        "source_dag_hash": canonical_evidence_hash(ancestry),
        "collar_layer": layer,
        "collar_layer_hash": canonical_evidence_hash(layer),
        "retained_densities": densities,
        "family_laws": family_laws,
        "retained_family_hash": canonical_evidence_hash(
            {"densities": densities, "family_laws": family_laws}
        ),
        "state_side_diagnostics": diagnostics or {},
    }


def _rehash_packet(packet: dict, coefficients: list[float]) -> None:
    layer = packet["collar_layer"]
    witness = packet["retained_densities"][-1]["flux_factorization"]
    witness_core = {
        "flux_basis_hash": canonical_evidence_hash(layer["flux_basis"]),
        "flux_coefficients": coefficients,
        "source_derivation_ids": witness["source_derivation_ids"],
        "refinement_natural": witness["refinement_natural"],
    }
    packet["collar_layer_hash"] = canonical_evidence_hash(layer)
    packet["retained_densities"][-1]["flux_factorization"] = {
        **witness_core,
        "witness_hash": canonical_evidence_hash(witness_core),
    }
    packet["retained_family_hash"] = canonical_evidence_hash(
        {
            "densities": packet["retained_densities"],
            "family_laws": packet["family_laws"],
        }
    )


def test_explicit_nonvacuous_flux_factorization_emits_packet_consistency_only() -> None:
    report = verify_collar_clause_packet(_packet())

    assert report["passed"] is False
    assert report["source_eligible"] is False
    assert report["COLLAR_CLAUSE_PACKET_CONSISTENCY_RECEIPT"] is True
    assert report["INDEPENDENT_COLLAR_RUN_REPLAY_RECEIPT"] is False
    assert report["nonvacuous_cross_cut"] is True
    assert report["sector_direct_sum_independent"] is True
    assert report["flux_is_proper_ambient_sector"] is True
    assert (
        report["retained_object_semantics"]
        == "bounded_real_coordinate_vector_not_verified_density_operator"
    )
    assert report["all_cross_cut_factor_through_flux"] is True
    cross = next(row for row in report["density_results"] if row["cross_cut"])
    assert cross["computed_support"] == "CROSS_CUT"
    assert (
        cross["coordinate_semantics"]
        == "bounded_real_coordinate_vector_not_verified_density_operator"
    )
    assert cross["factorization_residual"] == pytest.approx(0.0)
    assert cross["factorization_source_derived"] is True
    assert report["blockers"] == [
        "independent_collar_run_artifact_resolution_unavailable"
    ]


def test_T0_gibbs_entropy_and_cmi_do_not_promote_nonflux_cross_cut() -> None:
    packet = _packet(
        cross_coordinates=[0.0, 0.0, 1.0, 1.0],
        include_factorization=False,
        diagnostics={
            "density_matrices_valid": True,
            "gibbs_family_realized": True,
            "relative_entropy_nonnegative": True,
            "cmi_zero_or_bounded": True,
        },
    )

    report = verify_collar_clause_packet(packet)

    assert report["passed"] is False
    assert report["diagnostics_used_for_promotion"] is False
    assert "cross_cut_density_lacks_explicit_flux_factorization" in report["blockers"]
    assert report["mandatory_negative_controls"]["T0_STATE_SIDE"]["rejected"] is True


def test_T1_conditional_expectation_deselection_does_not_promote() -> None:
    packet = _packet(
        cross_coordinates=[0.0, 0.0, 1.0, 1.0],
        include_factorization=False,
        diagnostics={
            "conditional_expectation_exists": True,
            "conditional_expectation_kills_nonflux": True,
        },
    )

    report = verify_collar_clause_packet(packet)

    assert report["passed"] is False
    assert report["state_side_diagnostics"]["conditional_expectation_exists"] is True
    assert report["mandatory_negative_controls"]["T1_EFLUX"]["would_promote"] is False


def test_T2_modular_or_diagonal_membership_does_not_promote() -> None:
    packet = _packet(
        cross_coordinates=[0.0, 0.0, 1.0, 1.0],
        include_factorization=False,
        diagnostics={
            "modular_centralizer_contains_retained": True,
            "modular_diagonal_constraint": True,
        },
    )

    report = verify_collar_clause_packet(packet)

    assert report["passed"] is False
    assert report["state_side_diagnostics"]["modular_diagonal_constraint"] is True
    assert report["mandatory_negative_controls"]["T2_MODULAR"]["rejected"] is True


def test_numerical_flux_membership_without_explicit_witness_fails_closed() -> None:
    report = verify_collar_clause_packet(_packet(include_factorization=False))

    cross = next(row for row in report["density_results"] if row["cross_cut"])
    assert cross["numerically_in_flux_span"] is True
    assert cross["explicit_factorization_present"] is False
    assert report["passed"] is False


def test_forbidden_source_ancestry_blocks_an_algebraically_valid_packet() -> None:
    report = verify_collar_clause_packet(_packet(source_kind="fit"))

    assert report["source_dag_clean"] is False
    assert report["passed"] is False
    assert "source_dag_has_non_source_or_empty_ancestry" in report["blockers"]


def test_placeholder_source_digest_is_rejected_before_a_receipt() -> None:
    packet = _packet()
    packet["source_ancestry"][0]["sha256"] = "sha256:" + "0" * 64
    packet["source_dag_hash"] = canonical_evidence_hash(packet["source_ancestry"])

    with pytest.raises(ValueError, match="all-zero placeholder"):
        verify_collar_clause_packet(packet)


def test_tampered_factorization_is_recomputed_and_rejected() -> None:
    packet = _packet()
    packet["retained_densities"][-1]["flux_factorization"]["flux_coefficients"][0] = 7.0
    packet["retained_family_hash"] = canonical_evidence_hash(
        {
            "densities": packet["retained_densities"],
            "family_laws": packet["family_laws"],
        }
    )

    report = verify_collar_clause_packet(packet)

    cross = next(row for row in report["density_results"] if row["cross_cut"])
    assert cross["factorization_hash_matches"] is False
    assert cross["factorization_residual"] > 1.0
    assert report["passed"] is False


def test_receipt_validates_against_strict_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(verify_collar_clause_packet(_packet()))
    Draft202012Validator(schema).validate(
        verify_collar_clause_packet(
            _packet(
                cross_coordinates=[0.0, 0.0, 1.0, 1.0],
                include_factorization=False,
            )
        )
    )


def test_schema_binds_packet_consistency_to_recomputed_prerequisites() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    report = verify_collar_clause_packet(_packet())
    report["source_dag_hash_matches"] = False

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(report)


def test_negative_controls_and_lazy_exports_are_available() -> None:
    from oph_fpe import cosmology

    controls = collar_clause_negative_controls()
    assert set(controls) == {"T0_STATE_SIDE", "T1_EFLUX", "T2_MODULAR"}
    assert cosmology.verify_collar_clause_packet is verify_collar_clause_packet
    assert cosmology.collar_clause_negative_controls is collar_clause_negative_controls
    assert cosmology.canonical_collar_evidence_hash is canonical_evidence_hash


def test_malformed_unbounded_layer_emits_no_receipt() -> None:
    packet = _packet()
    packet["collar_layer"]["ambient_dimension"] = 257

    with pytest.raises(ValueError, match="ambient_dimension"):
        verify_collar_clause_packet(packet)

    with pytest.raises(ValueError, match="tolerance"):
        verify_collar_clause_packet(_packet(), tolerance=10**400)


def test_ill_conditioned_flux_basis_cannot_split_span_and_factorization_checks() -> None:
    packet = _packet()
    left = [
        -0.6627759336613024,
        0.6383090803039063,
        -0.3898937676369718,
        -0.03567113285021351,
    ]
    right = [
        0.6225461872098517,
        0.3535614821435405,
        -0.521861749173695,
        0.4637788674404114,
    ]
    flux = [
        [
            -27051217.797555614,
            -68122734.6063419,
            -66744443.88371733,
            13141683.91327499,
        ],
        [
            -27051217.797239408,
            -68122734.60628286,
            -66744443.88407811,
            13141683.912399568,
        ],
    ]
    packet["collar_layer"]["left_basis"] = [left]
    packet["collar_layer"]["right_basis"] = [right]
    packet["collar_layer"]["flux_basis"] = flux
    packet["retained_densities"][0]["coordinates"] = left
    packet["retained_densities"][1]["coordinates"] = right
    packet["retained_densities"][2]["coordinates"] = [
        0.00031620636582374573,
        0.000059038400650024414,
        -0.00036077946424484253,
        -0.0008754227310419083,
    ]
    _rehash_packet(packet, [-1.0, 1.0])

    report = verify_collar_clause_packet(packet)
    cross = next(row for row in report["density_results"] if row["cross_cut"])

    assert cross["factorization_residual"] == pytest.approx(0.0)
    assert cross["numerically_in_flux_span"] is False
    assert report["basis_numerical_conditioning_passed"] is False
    assert "basis_numerical_conditioning_out_of_bounds" in report["blockers"]
    assert report["all_cross_cut_factor_through_flux"] is False
    assert report["COLLAR_CLAUSE_PACKET_CONSISTENCY_RECEIPT"] is False


def test_overcomplete_sector_declaration_is_rejected_before_decomposition() -> None:
    packet = _packet()
    packet["collar_layer"]["flux_basis"] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _rehash_packet(packet, [2.0, -1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="sector ranks exceed"):
        verify_collar_clause_packet(packet)


def test_made_up_lean_theorem_digest_cannot_earn_source_receipt() -> None:
    packet = _packet()
    packet["source_ancestry"][0]["sha256"] = _content_hash("invented theorem")
    packet["source_dag_hash"] = canonical_evidence_hash(packet["source_ancestry"])

    report = verify_collar_clause_packet(packet)

    assert report["passed"] is False
    assert report["source_eligible"] is False
    assert report["pinned_theory_registry_complete"] is False
    assert "pinned_collar_theorem_registry_incomplete" in report["blockers"]
    assert "unregistered_or_mismatched_lean_theorem_artifact" in report["blockers"]


def test_flux_overlap_with_one_sided_sector_fails_direct_sum_gate() -> None:
    packet = _packet(cross_coordinates=[1.0, 0.0, 0.0, 0.0])
    packet["collar_layer"]["flux_basis"] = [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    _rehash_packet(packet, [0.0, 1.0])

    report = verify_collar_clause_packet(packet)

    assert report["passed"] is False
    assert report["sector_direct_sum_independent"] is False
    assert (
        "left_right_flux_sectors_not_direct_sum_independent" in report["blockers"]
    )


@pytest.mark.parametrize(
    "coordinates",
    (
        [True, 0.0, 0.0, 0.0],
        [float("inf"), 0.0, 0.0, 0.0],
        [1.0e13, 0.0, 0.0, 0.0],
        [10**400, 0.0, 0.0, 0.0],
    ),
)
def test_invalid_or_boolean_coordinate_vectors_are_rejected(
    coordinates: list[float],
) -> None:
    packet = _packet()
    packet["retained_densities"][0]["coordinates"] = coordinates
    packet["retained_family_hash"] = canonical_evidence_hash(
        {
            "densities": packet["retained_densities"],
            "family_laws": packet["family_laws"],
        }
    )

    with pytest.raises(ValueError, match="finite numeric vector"):
        verify_collar_clause_packet(packet)


def test_packet_byte_and_identifier_bounds_are_enforced() -> None:
    packet = _packet()
    packet["packet_id"] = "x" * 257
    with pytest.raises(ValueError, match="bounded string"):
        verify_collar_clause_packet(packet)

    packet = _packet()
    packet["state_side_diagnostics"] = {"oversized": "x" * 2_000_001}
    with pytest.raises(ValueError, match="bounded byte limit"):
        verify_collar_clause_packet(packet)

    packet = _packet()
    packet["packet_id"] = "\ud800"
    with pytest.raises(ValueError, match="bounded canonical evidence data"):
        verify_collar_clause_packet(packet)


def test_standalone_writer_persists_schema_valid_receipt(tmp_path: Path) -> None:
    packet = _packet()
    report = write_collar_clause_certificate(tmp_path, packet)
    persisted = json.loads(
        (tmp_path / "collar_clause_certificate.json").read_text(encoding="utf-8")
    )
    replay_packet = json.loads(
        (tmp_path / "collar_clause_input_packet.json").read_text(encoding="utf-8")
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert persisted == report
    assert replay_packet == packet
    assert canonical_evidence_hash(replay_packet) == report["packet_hash"]
    Draft202012Validator(schema).validate(report)
