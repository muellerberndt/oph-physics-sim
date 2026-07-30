from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from oph_fpe.gauge.seam_equalizer_current_control import (
    RANK_PRIMES,
    STATUS,
    ZERO_SUM_DIMENSION,
    build_report,
    exact_rational_rank,
    first_nonzero_commutators,
    metric_self_adjoint,
    metric_skew,
    modular_rank,
    render_report,
    seam_disagreement_generators,
    verify_report,
    zero_sum_gram_matrix,
)


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "repair_closure"
    / "seam_equalizer_current_control_report.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report()


def test_thirty_equalizers_act_self_adjointly_on_the_zero_sum_space() -> None:
    generators = seam_disagreement_generators()
    gram = zero_sum_gram_matrix()
    assert len(generators) == 30
    assert all(
        len(matrix) == ZERO_SUM_DIMENSION
        and all(len(row) == ZERO_SUM_DIMENSION for row in matrix)
        for matrix in generators
    )
    assert exact_rational_rank(generators) == 30
    assert all(metric_self_adjoint(matrix, gram) for matrix in generators)


def test_first_commutators_have_exact_rank_49() -> None:
    first = first_nonzero_commutators()
    gram = zero_sum_gram_matrix()
    assert len(first) == 120
    assert exact_rational_rank(first) == 49
    assert {
        modular_rank(first, prime)
        for prime in RANK_PRIMES
    } == {49}
    assert all(metric_skew(matrix, gram) for matrix in first)


def test_lie_and_operator_closures_exclude_the_12d_identification(
    report: dict,
) -> None:
    assert report["status"] == STATUS
    assert report["compact_lie_closure"][
        "modular_basis_round_ranks"
    ] == [49, 55, 55]
    assert report["compact_lie_closure"][
        "exact_rational_rank_of_witness_basis"
    ] == 55
    assert report["compact_lie_closure"][
        "identified_algebra"
    ] == "so(V_0,H), hence the compact real algebra so(11)"
    assert report["generated_operator_algebra"][
        "modular_basis_round_ranks"
    ] == [31, 110, 121]
    assert report["generated_operator_algebra"][
        "exact_rational_rank_of_witness_basis"
    ] == 121
    assert report["generated_operator_algebra"][
        "identified_algebra"
    ] == "End_Q(V_0), extending over R to all 11x11 operators"
    control = report["current_identification_control"]
    assert control["comparison_target_dimension"] == 12
    assert control["comparison_used_to_construct_generators_or_ranks"] is False
    assert control[
        "repair_equalizers_are_the_desired_12d_compact_current"
    ] is False


def test_receipts_are_fail_closed_at_the_current_identification(
    report: dict,
) -> None:
    assert report["receipts"] == {
        "SEAM_EQUALIZER_FIRST_COMMUTATOR_RANK_49_RECEIPT": True,
        "SEAM_EQUALIZER_SO11_CLOSURE_RECEIPT": True,
        "SEAM_EQUALIZER_FULL_OPERATOR_ALGEBRA_RECEIPT": True,
        "SEAM_EQUALIZER_IS_12D_COMPACT_CURRENT_RECEIPT": False,
    }
    assert report["source"]["target_fields_read_by_rank_engine"] == []
    assert report["source"]["laboratory_measurements_read"] is False
    assert "not a falsification of OPH" in report["claim_boundary"]


def test_stored_report_is_byte_exact_and_replays(report: dict) -> None:
    assert REPORT_PATH.read_text(encoding="utf-8") == render_report(report)
    stored = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert verify_report(stored)["passed"] is True


def test_tampered_report_is_rejected(report: dict) -> None:
    mutant = copy.deepcopy(report)
    mutant["first_commutators"]["exact_rational_rank"] = 12
    mutant["receipts"][
        "SEAM_EQUALIZER_FIRST_COMMUTATOR_RANK_49_RECEIPT"
    ] = False
    assert verify_report(mutant) == {
        "schema": "oph.seam-equalizer-current-control-verification/1.0.0",
        "passed": False,
        "expected_report_sha256": report["report_sha256"],
        "supplied_report_sha256": report["report_sha256"],
        "reason": "supplied report differs from exact recomputation",
    }
