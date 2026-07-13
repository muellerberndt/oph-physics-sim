from __future__ import annotations

import numpy as np

from oph_fpe.algebra.maxent_refinement import (
    gibbs_state,
    maxent_refinement_closure_report,
)
from oph_fpe.ensembles.quotient_ensemble import (
    fail_closed_promotion_receipts,
    rg_exponential_family_closure_receipt,
)


IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)


def _site_operator(operator: np.ndarray, site: int, site_count: int) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index in range(site_count):
        result = np.kron(result, operator if index == site else IDENTITY)
    return result


def _global_constraints(site_count: int) -> list[np.ndarray]:
    zz = sum(
        _site_operator(PAULI_Z, site, site_count)
        @ _site_operator(PAULI_Z, (site + 1) % site_count, site_count)
        for site in range(site_count)
    )
    transverse = sum(_site_operator(PAULI_X, site, site_count) for site in range(site_count))
    return [zz, transverse]


def _decimate_odd_sites(state: np.ndarray) -> np.ndarray:
    site_count = int(round(np.log2(state.shape[0])))
    tensor = state.reshape([2] * (2 * site_count))
    for site in reversed(range(1, site_count, 2)):
        tensor = np.trace(tensor, axis1=site, axis2=site + tensor.ndim // 2)
    coarse_count = site_count // 2
    return tensor.reshape(2**coarse_count, 2**coarse_count)


def _refinement_report(multipliers: np.ndarray):
    fine_constraints = _global_constraints(6)
    coarse_constraints = _global_constraints(3)
    fine_state, _ = gibbs_state(fine_constraints, multipliers)
    return maxent_refinement_closure_report(
        fine_state,
        _decimate_odd_sites,
        fine_constraints,
        coarse_constraints,
        refinement_channel_id="spin-ring-decimation-even-sites-v1",
    )


def test_closed_subfamily_emits_recomputed_rg_receipt() -> None:
    result = _refinement_report(np.array([0.0, 0.4]))
    evidence = result.as_jsonable()
    gate = rg_exponential_family_closure_receipt(evidence)

    assert result.closure_defect_nats < 1.0e-10
    assert result.trace_norm_residual <= result.pinsker_residual_bound + 1.0e-9
    assert result.counts_match_displayed_dimension is True
    assert evidence["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is True
    assert gate["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is True

    fail_closed = fail_closed_promotion_receipts(
        claim_tier="E2",
        baseline_kind="finite_test",
        rg_closure_evidence=evidence,
    )
    assert fail_closed["receipts"]["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is True
    assert "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT" in fail_closed["claim_tier_gate"][
        "required_receipts"
    ]


def test_generic_refinement_quantifies_nonclosure_without_promoting() -> None:
    result = _refinement_report(np.array([0.7, 0.4]))
    evidence = result.as_jsonable()

    assert result.projection_converged is True
    assert result.moment_residual_linf < 1.0e-9
    assert result.duhamel_hessian_min_eigenvalue > 1.0e-9
    assert result.closure_defect_nats > 1.0e-6
    assert result.trace_norm_residual <= result.pinsker_residual_bound + 1.0e-9
    assert evidence["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is False
    assert rg_exponential_family_closure_receipt(evidence)[
        "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"
    ] is False


def test_rg_gate_ignores_tampered_top_level_boolean() -> None:
    evidence = _refinement_report(np.array([0.7, 0.4])).as_jsonable()
    evidence["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] = True

    gate = rg_exponential_family_closure_receipt(evidence)

    assert gate["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is False
    assert "exponential_family_closure_defect_nonzero" in gate["promotion_blockers"]


def test_rg_gate_requires_actual_matrix_evidence() -> None:
    gate = rg_exponential_family_closure_receipt(None)

    assert gate["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"] is False
    assert gate["promotion_blockers"] == ["finite_matrix_refinement_evidence_missing"]
