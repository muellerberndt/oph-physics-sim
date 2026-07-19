from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from oph_fpe.cosmology.collar_clause import (
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
    source_kind: str = "lean_theorem",
) -> dict:
    ancestry = [
        {
            "node_id": "collar-positive-theorem",
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
            "source_derivation_ids": ["collar-positive-theorem"],
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
        "source_derivation_ids": ["collar-positive-theorem"],
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


def test_explicit_nonvacuous_flux_factorization_emits_source_receipt() -> None:
    report = verify_collar_clause_packet(_packet())

    assert report["passed"] is True
    assert report["source_eligible"] is True
    assert report["nonvacuous_cross_cut"] is True
    assert report["all_cross_cut_factor_through_flux"] is True
    cross = next(row for row in report["density_results"] if row["cross_cut"])
    assert cross["computed_support"] == "CROSS_CUT"
    assert cross["factorization_residual"] == pytest.approx(0.0)
    assert cross["factorization_source_derived"] is True
    assert report["blockers"] == []


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


def test_standalone_writer_persists_schema_valid_receipt(tmp_path: Path) -> None:
    report = write_collar_clause_certificate(tmp_path, _packet())
    persisted = json.loads(
        (tmp_path / "collar_clause_certificate.json").read_text(encoding="utf-8")
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert persisted == report
    Draft202012Validator(schema).validate(report)
