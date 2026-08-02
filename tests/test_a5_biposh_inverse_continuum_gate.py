from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.cosmology.a5_biposh_inverse_continuum_gate import (
    DEFAULT_RECEIPT,
    SCHEMA,
    STATUS,
    _canonical_bytes,
    build_inverse_continuum_gate_packet,
    full_raw_stiffness_tail,
)
from oph_fpe.cosmology.verify_a5_biposh_inverse_continuum_gate_independent import (
    VerificationError,
    verify_packet,
)


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))


def _write_rehashed(tmp_path: Path, packet: dict) -> Path:
    packet.pop("payload_sha256", None)
    packet["payload_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(packet)
    ).hexdigest()
    path = tmp_path / "mutated_inverse_gate.json"
    path.write_bytes(_canonical_bytes(packet))
    return path


def test_full_raw_tail_covers_every_harmonic_block() -> None:
    tail = full_raw_stiffness_tail()
    assert tail["raw_harmonic_dimension"] == 77
    assert len(tail["block_rows"]) == 49
    assert tail["full_raw_stiffness_cauchy_limit_exists"] is True
    assert tail["md_geometric_ratio_upper_bound"]["binary64_upper"] < 1.0
    assert tail["mm_geometric_ratio_upper_bound"]["binary64_upper"] < 1.0
    assert tail[
        "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
    ]["binary64_upper"] > 0.0
    assert tail["operator_tail_bound_is_exact_rational_upper_arithmetic"] is True


def test_committed_packet_rebuilds_exactly(canonical: dict) -> None:
    rebuilt = build_inverse_continuum_gate_packet()
    assert rebuilt == canonical
    assert rebuilt["schema"] == SCHEMA
    assert rebuilt["status"] == STATUS


def test_correlated_kernel_verifier_reconstructs_packet(canonical: dict) -> None:
    verified = verify_packet()
    assert verified["payload_sha256"] == canonical["payload_sha256"]
    assert verified["verification_scope"] == {
        "exact_tail_arithmetic_reimplemented": True,
        "registered_mesh_builder_shared": True,
        "harmonic_design_stiffness_and_biposh_kernels_shared": True,
        "independent_harmonic_implementation": False,
        "classification": (
            "correlated-kernel replay with separately implemented certificate "
            "and boundary checks"
        ),
    }


def test_inverse_gate_fails_without_relabeling_failure_as_no_go(
    canonical: dict,
) -> None:
    gate = canonical["inverse_admission_gate"]
    assert gate["finite_anchor_neumann_gate_epsilon_lt_gap"] is False
    assert gate["finite_anchor_gap_is_continuum_coercivity_certificate"] is False
    assert gate["full_raw_inverse_continuum_tail_certified"] is False
    assert gate["projected_inverse_continuum_tail_certified"] is False
    assert canonical["selection_decision"]["full_raw_stiffness_cauchy_limit"] is True
    assert canonical["selection_decision"]["uniform_continuum_coercivity"] is False


def test_operational_response_is_kept_separate_from_physical_prediction(
    canonical: dict,
) -> None:
    response = canonical["operational_stiffness_response"]
    assert response["readback_identity"] == "L = 2*edge_count*(I-R)"
    assert response["stiffness_statistic_can_be_an_operational_response_observable"] is True
    assert response[
        "declared_registered_ladder_primitive_alphabet_source_emitted"
    ] is True
    assert response[
        "declared_registered_ladder_unit_counting_source_emitted"
    ] is True
    assert response["declared_registered_geometry_levels"] == [0, 1, 2, 3, 4, 5]
    assert response["declared_ladder_reaches_inverse_anchor_level"] is False
    assert response["first_order_refinement_readback_discharged"] is True
    assert response["full_refinement_commuting_diagram_discharged"] is False
    assert response["physical_repair_law_selected"] is False
    assert response["all_level_response_law_source_selected"] is False
    assert response["response_readout_physically_attached"] is False
    assert response["stiffness_statistic_is_a_current_physical_prediction"] is False
    assert canonical["selection_decision"]["physical_prediction"] is False
    assert canonical["selection_decision"]["promotion_allowed"] is False


def test_copy_space_counterexample_preserves_transfer_boundary(canonical: dict) -> None:
    example = canonical["exact_counterexamples"][
        "multiplicity_space_radial_mixing"
    ]
    assert example["first_statistic"] == 1.0
    assert example["mixed_statistic"] == 0.5
    assert example["both_copy_space_readouts_commute_with_spatial_rotations"] is True
    assert example["rotation_equivariance_forces_scalar_on_copy_space"] is False
    transfer = canonical["transfer_boundary"]
    assert transfer["scalar_rescaling_cancellation_proved"] is True
    assert transfer["rotation_equivariant_transfer_proved"] is False
    assert transfer["multiplicity_one_proved"] is False
    assert transfer["radial_copy_mixing_excluded"] is False


def test_shape_regular_coercivity_route_and_anisotropic_limit_stay_typed(
    canonical: dict,
) -> None:
    geometry = canonical["continuum_geometry_assessment"]
    shape = geometry["finite_shape_regular_diagnostic"]
    assert len(shape["rows"]) == 8
    assert shape["all_level_uniform_shape_regularity_proved"] is False
    comparison = geometry["equal_counting_vs_cotangent_diagnostic"]
    assert comparison["equal_counting_is_the_declared_repair_measure"] is True
    assert comparison["cotangent_weights_are_the_declared_repair_measure"] is False
    assert comparison["finite_binary64_contrast_is_a_continuum_proof"] is False
    assert comparison["rows"][-1]["equal_counting_primary_statistic"] > 0.03
    assert comparison["rows"][-1]["cotangent_fem_primary_statistic"] < 1.0e-5
    route = geometry["shape_regular_coercivity_route"]
    assert route["candidate_route_identified"] is True
    assert route["not_excluded_by_current_results"] is True
    assert route["closed_here"] is False
    assert route["does_not_force_so3_isotropy_or_zero_l6"] is True


def test_rehashed_inverse_promotion_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["selection_decision"]["full_inverse_covariance_continuum_limit"] = True
    mutated["selection_decision"]["physical_prediction"] = True
    with pytest.raises(VerificationError, match="selection boundary"):
        verify_packet(_write_rehashed(tmp_path, mutated))


def test_rehashed_radial_mixing_erasure_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["transfer_boundary"]["radial_copy_mixing_excluded"] = True
    with pytest.raises(VerificationError, match="transfer boundary"):
        verify_packet(_write_rehashed(tmp_path, mutated))


def test_rehashed_tail_reduction_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["full_raw_stiffness_tail"][
        "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
    ]["numerator"] //= 10
    with pytest.raises(VerificationError, match="Cauchy flag"):
        verify_packet(_write_rehashed(tmp_path, mutated))


def test_rehashed_downward_binary64_rational_view_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    mutated = copy.deepcopy(canonical)
    rational = mutated["full_raw_stiffness_tail"][
        "maximum_edge_upper_bound_radians"
    ]
    rational["binary64_upper"] = 0.0
    with pytest.raises(VerificationError, match="rational upper view"):
        verify_packet(_write_rehashed(tmp_path, mutated))


def test_rehashed_lean_theorem_name_drift_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["lean_boundary"]["inverse_counterexample_theorems"][1] = (
        "collapsing_stiffness_converges"
    )
    with pytest.raises(VerificationError, match="Lean declaration boundary"):
        verify_packet(_write_rehashed(tmp_path, mutated))
