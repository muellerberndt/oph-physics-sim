from oph_fpe.ensembles.quotient_ensemble import (
    canonicalize_representative,
    claim_tier_gate,
    quotient_lumpability_receipt,
    representative_lift_firewall_receipt,
    sampler_correctness_receipt,
)


def test_canonicalizer_drops_executor_metadata() -> None:
    a = {
        "visible_record": {"packet": "alpha", "value": 3},
        "worker_id": "w0",
        "queue_order": 7,
        "hidden_carrier_coordinates": [1.0, 2.0],
    }
    b = {
        "queue_order": 8,
        "hidden_carrier_coordinates": [9.0],
        "visible_record": {"value": 3, "packet": "alpha"},
        "worker_id": "w1",
    }

    assert canonicalize_representative(a)["quotient_key"] == canonicalize_representative(b)["quotient_key"]


def test_e1_claim_tier_gate_fails_closed() -> None:
    gate = claim_tier_gate(claim_tier="E1", receipts={})

    assert gate["promotion_allowed"] is False
    assert "E1_is_not_a_physical_promotion_tier" in gate["promotion_blockers"]


def test_representative_lift_firewall_blocks_undeclared_orbit_bias() -> None:
    receipt = representative_lift_firewall_receipt(
        uniform_representative_sampling_declared=True,
        representative_counting_declared_as_physical=False,
        orbit_size_correction_applied=False,
        max_orbit_bias=0.25,
    )

    assert receipt["REPRESENTATIVE_LIFT_FIREWALL_RECEIPT"] is False
    assert receipt["representative_lift_check"] == "fail"


def test_exact_lumpability_and_sampler_receipts_can_pass_for_toy_kernel() -> None:
    lump = quotient_lumpability_receipt(
        max_lumpability_residual=0.0,
        representative_kernel_hash="rep",
        quotient_kernel_hash="quo",
    )
    sampler = sampler_correctness_receipt(
        sampler_type="metropolis_hastings",
        target_weight_hash="w",
        proposal_kernel_hash="r",
        acceptance_rule_hash="mh",
        hastings_asymmetry_included=True,
        detailed_balance_max_error=0.0,
        stationarity_tv_error=0.0,
        irreducible=True,
        aperiodic=True,
    )

    assert lump["QUOTIENT_LUMPABILITY_RECEIPT"] is True
    assert sampler["SAMPLER_CORRECTNESS_RECEIPT"] is True
