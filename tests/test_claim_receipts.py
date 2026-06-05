import numpy as np

from oph_fpe.claims import BRANCH_INSTANTIATION_SANITY, checked_claim_level
from oph_fpe.consensus.boundary_fiber import boundary_conditioned_uniqueness_receipt
from oph_fpe.consensus.fair_block import fair_block_consensus_certificate
from oph_fpe.consensus.lyapunov import lyapunov_descent_receipt
from oph_fpe.gauge.mar_sieve import standard_model_candidate_sieve
from oph_fpe.gauge.repair_projection import exact_repair_projection_receipt


def test_claim_level_registry_includes_bw_branch_sanity():
    assert checked_claim_level(BRANCH_INSTANTIATION_SANITY) == BRANCH_INSTANTIATION_SANITY


def test_lyapunov_descent_receipt_uses_trace_phi_pairs():
    report = lyapunov_descent_receipt([
        {"phi_before": 10, "phi": 7},
        {"phi_before": 7, "phi": 7},
        {"phi_before": 7, "phi": 0},
    ])

    assert report["LYAPUNOV_DESCENT_RECEIPT"] is True
    assert report["claim_level"] == "recovered_core"
    assert report["strict_descent_steps"] == 2


def test_boundary_fiber_receipt_requires_unique_extension():
    report = boundary_conditioned_uniqueness_receipt(
        boundary_map_preserved=True,
        sector_map_preserved=True,
        consistent_extension_count=1,
        checked_states=8,
    )

    assert report["BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT"] is True
    assert report["claim_level"] == "recovered_core"


def test_exact_repair_projection_receipt_rejects_non_projection():
    projection = np.diag([1.0, 0.0, 1.0])
    good = exact_repair_projection_receipt(projection)
    bad = exact_repair_projection_receipt(np.array([[1.0, 0.3], [0.0, 1.0]]))

    assert good["EXACT_REPAIR_EQUALS_PROJECTION_RECEIPT"] is True
    assert bad["EXACT_REPAIR_EQUALS_PROJECTION_RECEIPT"] is False


def test_sm_candidate_sieve_checks_visible_low_energy_gates():
    report = standard_model_candidate_sieve({
        "G_phys": "(SU(3)xSU(2)xU(1))/Z6",
        "hypercharge_lattice": "exact",
        "Nc": 3,
        "Ng": 3,
        "higgs_doublets": 1,
        "light_chiral_exotics": 0,
        "extra_low_scale_u1": 0,
        "xy_gauge_bosons": 0,
    })

    assert report["SM_QUOTIENT_GATE_RECEIPT"] is True
    assert report["claim_level"] == "continuation"


def test_fair_block_certificate_is_conditional_continuation():
    report = fair_block_consensus_certificate(
        lambda_contraction=0.8,
        epsilon_noise=0.01,
        beta=2.0,
        lipschitz_L=1.5,
        block_count=12,
        active_fraction=0.75,
    )

    assert report["FAIR_BLOCK_CONSENSUS_CERTIFICATE"] is True
    assert report["claim_level"] == "continuation"
