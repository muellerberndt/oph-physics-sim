from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations
from oph_fpe.modular_gearing import (
    OrientedRepairEdge,
    OrientedTransitionLedger,
    PhysicalSourceEvidence,
    PresentationTransitionRate,
    a5_oriented_spectrum_diagnostics,
    build_modular_gearing_receipt,
    canonical_a5_slot_permutations,
    compress_modular_channels,
    entropy_production_diagnostics,
    fundamental_cycle_holonomy,
    modular_gearing_firewall,
    oriented_slot_names,
    raw_24_channel_realization,
    strong_quotient_lumpability_evidence,
    validate_transition_ledger,
    weighted_hodge_reconstruction,
)


SOURCE_HASH = "sha256:" + "a" * 64


def _paired_edge(
    port: int,
    source: str,
    target: str,
    q_plus: int | Fraction,
    q_minus: int | Fraction,
    *,
    prefix: str | None = None,
    semantics: str = "synthetic_fixture_rate",
    source_hash: str | None = None,
) -> tuple[OrientedRepairEdge, OrientedRepairEdge]:
    stem = prefix or f"e{port}"
    plus = OrientedRepairEdge(
        f"{stem}+",
        source,
        target,
        f"{stem}-",
        port,
        "+",
        q_plus,
        semantics,
        source_dag_hash=source_hash,
    )
    minus = OrientedRepairEdge(
        f"{stem}-",
        target,
        source,
        f"{stem}+",
        port,
        "-",
        q_minus,
        semantics,
        source_dag_hash=source_hash,
    )
    return plus, minus


def _a5_fixture(*, physical_semantics: bool = False):
    semantics = (
        "aggregate_quotient_continuous_time_rate"
        if physical_semantics
        else "synthetic_fixture_rate"
    )
    source_hash = SOURCE_HASH if physical_semantics else None
    states = tuple(
        [f"x{port}" for port in range(12)] + [f"y{port}" for port in range(12)]
    )
    pairs = [
        _paired_edge(
            port,
            f"x{port}",
            f"y{port}",
            Fraction(1),
            Fraction(2),
            semantics=semantics,
            source_hash=source_hash,
        )
        for port in range(12)
    ]
    # Canonical order matches the two orientation layers.
    edges = tuple([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
    ledger = OrientedTransitionLedger(states, edges)
    mapping = {f"sigma-{state}": state for state in states}
    representative_rates = tuple(
        PresentationTransitionRate(
            f"sigma-{edge.source_quotient_state}",
            f"sigma-{edge.target_quotient_state}",
            edge.base_port_id,
            edge.orientation,
            edge.rate,
            semantics,
        )
        for edge in edges
    )
    lumpability = strong_quotient_lumpability_evidence(
        ledger, mapping, representative_rates
    )
    port_actions = icosahedral_a5_port_permutations()
    state_actions = tuple(
        tuple(list(row) + [12 + value for value in row]) for row in port_actions
    )
    edge_actions = canonical_a5_slot_permutations()
    realization = raw_24_channel_realization(ledger)
    return ledger, lumpability, realization, state_actions, edge_actions


def test_ledger_enforces_24_labels_and_reverse_pair_involution() -> None:
    plus, minus = _paired_edge(0, "x", "y", 2, 1)
    ledger = OrientedTransitionLedger(("x", "y"), (plus, minus))
    report = validate_transition_ledger(ledger)

    assert len(oriented_slot_names()) == 24
    assert report["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"] is True
    assert report["REVERSE_PAIR_INVOLUTION_RECEIPT"] is True

    broken = OrientedTransitionLedger(
        ("x", "y"), (plus, replace(minus, reverse_edge_id="missing"))
    )
    assert (
        validate_transition_ledger(broken)[
            "ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"
        ]
        is False
    )


@pytest.mark.parametrize(
    "forbidden_semantics",
    [
        "worker_attempt_count",
        "scheduler_order",
        "queue_position",
        "rejected_proposal_count",
    ],
)
def test_execution_metadata_is_never_accepted_as_a_rate(
    forbidden_semantics: str,
) -> None:
    plus, minus = _paired_edge(
        0,
        "x",
        "y",
        2,
        1,
        semantics=forbidden_semantics,
    )
    report = validate_transition_ledger(
        OrientedTransitionLedger(("x", "y"), (plus, minus))
    )

    assert report["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"] is False
    assert report["FORBIDDEN_EXECUTION_METADATA_USED_AS_RATES"] is True
    assert report["WORKER_ATTEMPTS_USED_AS_RATES"] is (
        forbidden_semantics == "worker_attempt_count"
    )


def test_strong_channelwise_lumpability_is_computed_and_matches_ledger() -> None:
    plus, minus = _paired_edge(0, "x", "y", Fraction(2), Fraction(1))
    ledger = OrientedTransitionLedger(("x", "y"), (plus, minus))
    mapping = {"x1": "x", "x2": "x", "y1": "y", "y2": "y"}
    rates = (
        PresentationTransitionRate("x1", "y1", 0, "+", 2, "synthetic_fixture_rate"),
        PresentationTransitionRate("x2", "y2", 0, "+", 2, "synthetic_fixture_rate"),
        PresentationTransitionRate("y1", "x1", 0, "-", 1, "synthetic_fixture_rate"),
        PresentationTransitionRate("y2", "x2", 0, "-", 1, "synthetic_fixture_rate"),
    )
    evidence = strong_quotient_lumpability_evidence(ledger, mapping, rates)

    assert evidence.passed is True
    assert evidence.exact_strong_lumpability is True
    assert evidence.quotient_ledger_maximum_absolute_defect == 0.0

    failed = strong_quotient_lumpability_evidence(
        ledger,
        mapping,
        (replace(rates[0], rate=3),) + rates[1:],
    )
    assert failed.passed is False
    assert failed.maximum_absolute_defect == 1.0


def test_exact_lumpability_does_not_round_distinct_large_integer_rates() -> None:
    huge = 2**54
    plus, minus = _paired_edge(0, "x", "y", huge, 1)
    ledger = OrientedTransitionLedger(("x", "y"), (plus, minus))
    mapping = {"x1": "x", "x2": "x", "y1": "y", "y2": "y"}
    rates = (
        PresentationTransitionRate(
            "x1", "y1", 0, "+", huge, "synthetic_fixture_rate"
        ),
        PresentationTransitionRate(
            "x2", "y2", 0, "+", huge + 1, "synthetic_fixture_rate"
        ),
        PresentationTransitionRate("y1", "x1", 0, "-", 1, "synthetic_fixture_rate"),
        PresentationTransitionRate("y2", "x2", 0, "-", 1, "synthetic_fixture_rate"),
    )

    evidence = strong_quotient_lumpability_evidence(
        ledger, mapping, rates, tolerance=0.0
    )

    assert evidence.exact_arithmetic_available is True
    assert evidence.exact_strong_lumpability is False
    assert evidence.maximum_absolute_defect == 1.0
    assert evidence.passed is False


def _triangle_ledger(chord_rate: int) -> OrientedTransitionLedger:
    xy = _paired_edge(0, "x", "y", 2, 1, prefix="xy")
    yz = _paired_edge(1, "y", "z", 2, 1, prefix="yz")
    xz = _paired_edge(2, "x", "z", chord_rate, 1, prefix="xz")
    return OrientedTransitionLedger(
        ("x", "y", "z"),
        (xy[0], xy[1], yz[0], yz[1], xz[0], xz[1]),
    )


def test_exact_fundamental_cycle_and_hodge_reconstruction() -> None:
    ledger = _triangle_ledger(4)
    holonomy = fundamental_cycle_holonomy(ledger)
    hodge = weighted_hodge_reconstruction(ledger, affinity_noise_weighted_bound=0.01)

    assert holonomy["exact_arithmetic_available"] is True
    assert holonomy["fundamental_cycle_count"] == 1
    assert holonomy["FUNDAMENTAL_CYCLE_HOLONOMY_RECEIPT"] is True
    assert holonomy["components"][0]["within_component_probability"] == pytest.approx(
        {"x": 1 / 7, "y": 2 / 7, "z": 4 / 7}
    )
    assert hodge["cycle_residual_linf"] < 1.0e-12
    assert hodge["weighted_laplacian_positive_spectral_gap"] > 0.0
    assert hodge["potential_l2_error_bound"] > 0.0

    failed = fundamental_cycle_holonomy(_triangle_ledger(3))
    failed_hodge = weighted_hodge_reconstruction(_triangle_ledger(3))
    assert failed["exact_fundamental_cycles_passed"] is False
    assert failed["FUNDAMENTAL_CYCLE_HOLONOMY_RECEIPT"] is False
    assert failed_hodge["cycle_residual_linf"] > 0.0
    assert failed_hodge["EXACT_INTEGRABLE_MODULAR_PART_RECEIPT"] is False


def test_disconnected_sector_weights_are_not_silently_selected() -> None:
    first = _paired_edge(0, "a", "b", 2, 1, prefix="ab")
    second = _paired_edge(1, "c", "d", 3, 1, prefix="cd")
    ledger = OrientedTransitionLedger(
        ("a", "b", "c", "d"),
        (first[0], first[1], second[0], second[1]),
    )
    holonomy = fundamental_cycle_holonomy(ledger)
    entropy = entropy_production_diagnostics(ledger)

    assert holonomy["component_count"] == 2
    assert holonomy["sector_weight_ambiguity_dimension"] == 1
    assert holonomy["GLOBAL_SECTOR_WEIGHT_SELECTION_RECEIPT"] is False
    assert entropy["global_stationary_law_unique"] is False
    assert entropy["global_entropy_production"] is None


def test_aggregate_detailed_balance_does_not_hide_channel_currents() -> None:
    channel_a = _paired_edge(0, "x", "y", 2, 1, prefix="a")
    channel_b = _paired_edge(1, "x", "y", 1, 2, prefix="b")
    ledger = OrientedTransitionLedger(
        ("x", "y"),
        (channel_a[0], channel_a[1], channel_b[0], channel_b[1]),
    )
    report = entropy_production_diagnostics(ledger)

    assert report["AGGREGATE_DETAILED_BALANCE_RECEIPT"] is True
    assert report["CHANNELWISE_DETAILED_BALANCE_RECEIPT"] is False
    assert report["components"][0]["aggregate_entropy_production"] == pytest.approx(0.0)
    assert report["components"][0]["channelwise_entropy_production"] > 0.0


def test_sparse_and_dense_channel_whitening_compute_same_omega() -> None:
    ledger, _, sparse_realization, _, _ = _a5_fixture()
    assert sparse.issparse(sparse_realization)
    sparse_result = compress_modular_channels(ledger, sparse_realization)
    dense_result = compress_modular_channels(ledger, sparse_realization.toarray())

    assert sparse_result.full_column_rank is True
    assert sparse_result.whitening_residual < 1.0e-12
    assert sparse_result.closure_residual_hs < 1.0e-12
    assert np.allclose(sparse_result.omega_24, dense_result.omega_24)

    rank_deficient = compress_modular_channels(
        ledger, np.zeros((len(ledger.edges), 24))
    )
    assert rank_deficient.full_column_rank is False
    assert rank_deficient.omega_24 is None


def test_raw_label_with_multiple_affinities_has_channel_leakage() -> None:
    ledger, _, _, _, _ = _a5_fixture()
    extra = _paired_edge(0, "u", "v", 1, 3, prefix="extra")
    extended = OrientedTransitionLedger(
        ledger.quotient_states + ("u", "v"), ledger.edges + extra
    )
    result = compress_modular_channels(extended, raw_24_channel_realization(extended))

    assert result.full_column_rank is True
    assert result.closure_residual_hs > 0.1


def test_exact_a5_orientation_classifier_finds_paired_1_3_3_5_spectrum() -> None:
    ledger, _, realization, state_actions, edge_actions = _a5_fixture()
    compression = compress_modular_channels(ledger, realization)
    result = a5_oriented_spectrum_diagnostics(
        ledger,
        compression,
        state_order=ledger.quotient_states,
        a5_state_permutations=state_actions,
        a5_edge_permutations=edge_actions,
    ).report()

    assert result["A5_ORIENTATION_COVARIANCE_RECEIPT"] is True
    assert result["A5_CHARACTER_PROJECTOR_DIMENSIONS_RECEIPT"] is True
    assert result["A5_ORIENTED_1_3_3_5_PAIRED_SPECTRUM_RECEIPT"] is True
    assert [row["each_sign_multiplicity"] for row in result["irrep_rows"]] == [
        1,
        3,
        3,
        5,
    ]
    assert all(row["omega"] == pytest.approx(np.log(2)) for row in result["irrep_rows"])

    corrupted = list(edge_actions)
    corrupted[1] = corrupted[0]
    failed = a5_oriented_spectrum_diagnostics(
        ledger,
        compression,
        state_order=ledger.quotient_states,
        a5_state_permutations=state_actions,
        a5_edge_permutations=corrupted,
    ).report()
    assert failed["edge_action_permutations_valid"] is False
    assert failed["A5_ORIENTED_1_3_3_5_PAIRED_SPECTRUM_RECEIPT"] is False


def test_claim_tiers_are_mathematically_useful_but_physically_fail_closed() -> None:
    ledger, lumpability, realization, state_actions, edge_actions = _a5_fixture()
    report = build_modular_gearing_receipt(
        ledger,
        lumpability=lumpability,
        channel_realization=realization,
        state_order=ledger.quotient_states,
        a5_state_permutations=state_actions,
        a5_edge_permutations=edge_actions,
    )

    assert report["MG0_CHANNEL_DIAGNOSTIC_RECEIPT"] is True
    assert report["MG1_FINITE_REVERSIBLE_GEARING_DIAGNOSTIC_RECEIPT"] is True
    assert report["MG2_24_CHANNEL_CARRIER_DIAGNOSTIC_RECEIPT"] is True
    assert report["MG3_A5_GEARED_SPECTRUM_DIAGNOSTIC_RECEIPT"] is True
    assert report["PHYSICAL_MG1_FINITE_REVERSIBLE_GEARING_RECEIPT"] is False
    assert report["PHYSICAL_MG3_A5_GEARED_SPECTRUM_RECEIPT"] is False
    assert report["MG4_OPERATIONALLY_CLOCKED_RECEIPT"] is False
    assert report["MG5_BW_PROMOTED_RECEIPT"] is False
    assert report["ELECTROWEAK_HIERARCHY_FROM_MODULAR_GEARING_RECEIPT"] is False


def test_plausible_hashes_and_filenames_cannot_promote_physical_mg_tiers() -> None:
    ledger, lumpability, realization, state_actions, edge_actions = _a5_fixture(
        physical_semantics=True
    )
    compression = compress_modular_channels(ledger, realization)
    symmetry = a5_oriented_spectrum_diagnostics(
        ledger,
        compression,
        state_order=ledger.quotient_states,
        a5_state_permutations=state_actions,
        a5_edge_permutations=edge_actions,
    )
    evidence = PhysicalSourceEvidence(
        source_dag_sha256=SOURCE_HASH,
        quotient_map_sha256="sha256:" + "b" * 64,
        rate_observation_bundle_sha256="sha256:" + "c" * 64,
        independent_verifier_receipt_sha256="sha256:" + "d" * 64,
        lumpability_evidence_sha256=lumpability.evidence_sha256,
        channel_realization_sha256=compression.realization_sha256,
        symmetry_action_sha256=symmetry.action_sha256,
        artifact_paths=("runs/quotient-map.json", "runs/rate-observations.json"),
    )
    report = build_modular_gearing_receipt(
        ledger,
        lumpability=lumpability,
        channel_realization=realization,
        state_order=ledger.quotient_states,
        a5_state_permutations=state_actions,
        a5_edge_permutations=edge_actions,
        physical_source_evidence=evidence,
    )

    assert report["PHYSICAL_MG1_FINITE_REVERSIBLE_GEARING_RECEIPT"] is False
    assert report["PHYSICAL_MG2_24_CHANNEL_CARRIER_RECEIPT"] is False
    assert report["PHYSICAL_MG3_A5_GEARED_SPECTRUM_RECEIPT"] is False
    assert "strict_common_source_tower_reverification_failed" in report[
        "physical_promotion_blockers"
    ]
    assert report["MG4_OPERATIONALLY_CLOCKED_RECEIPT"] is False
    assert report["MG5_BW_PROMOTED_RECEIPT"] is False


def test_firewall_preserves_all_pro_no_go_boundaries() -> None:
    firewall = modular_gearing_firewall()

    assert firewall["CENTRAL_REGISTER_MODULAR_FLOW_TRIVIAL_NO_GO"] is True
    assert (
        firewall["SYMMETRIC_IRREDUCIBLE_24_STATE_CHAIN_MODULAR_FLOW_TRIVIAL_NO_GO"]
        is True
    )
    assert firewall["NO_A5_CANONICAL_24_CYCLE_NO_GO"] is True
    assert firewall["REPAIR_GENERATOR_EQUALS_MODULAR_FLOW"] is False
    assert firewall["RATE_SCALE_DETERMINES_MODULAR_CLOCK"] is False
    assert firewall["CHANNEL_COUNT_24_DETERMINES_CLOCK"] is False
    assert firewall["DETERMINISTIC_NORMAL_FORM_SELECTS_PROBABILITY_LAW"] is False
