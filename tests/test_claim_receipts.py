import json

import numpy as np

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    CANONICAL_RECEIPTS,
    RECEIPT_SCHEMA_VERSION,
    checked_claim_level,
    with_claim_metadata,
)
from oph_fpe.consensus.boundary_fiber import boundary_conditioned_uniqueness_receipt
from oph_fpe.consensus.fair_block import fair_block_consensus_certificate
from oph_fpe.consensus.lyapunov import lyapunov_descent_receipt
from oph_fpe.gauge.higgs_carrier import borel_weil_higgs_carrier_receipt, write_borel_weil_higgs_carrier_report
from oph_fpe.gauge.mar_sieve import standard_model_candidate_sieve
from oph_fpe.gauge.repair_projection import exact_repair_projection_receipt


def test_claim_level_registry_includes_bw_branch_sanity():
    assert checked_claim_level(BRANCH_INSTANTIATION_SANITY) == BRANCH_INSTANTIATION_SANITY


def test_claim_metadata_writes_canonical_schema_fields():
    report = with_claim_metadata(
        {"mode": "unit_test_report"},
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt=CANONICAL_RECEIPTS["R2"],
        observable_id="unit_test_observable",
        fit_objective="unit_test_gate",
    )

    assert report["receipt_schema_version"] == RECEIPT_SCHEMA_VERSION
    assert report["receipt_name"] == "BW_KMS_BRANCH_REPLAY_RECEIPT"
    assert report["observable_id"] == "unit_test_observable"
    assert report["fit_objective"] == "unit_test_gate"
    assert report["physical_claim"] is False


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


def test_borel_weil_higgs_carrier_receipt_is_carrier_only():
    report = borel_weil_higgs_carrier_receipt()
    geometry = report["symmetry_breaking_geometry"]
    action = report["group_action_acceptance"]

    assert report["BOREL_WEIL_HIGGS_CARRIER_RECEIPT"] is True
    assert report["schema"] == "oph_borel_weil_higgs_carrier_bridge_v1"
    assert report["mode"] == "borel_weil_higgs_carrier_bridge_v1"
    assert report["checks"]["section_degree_is_minimal_nontrivial"] is True
    assert report["checks"]["neutral_lower_component"] is True
    assert report["checks"]["projective_ray_stabilizer_is_two_torus"] is True
    assert report["checks"]["vector_stabilizer_is_u1_q"] is True
    assert geometry["cover_action"] == "(g,z).phi = z^3 g phi"
    assert geometry["projective_ray_stabilizer_on_cover"] == "{(diag(a,a^-1),z): a,z in U(1)}"
    assert geometry["projective_ray_stabilizer"] == "(U(1)_T3 x U(1)_Y)/finite_center"
    assert geometry["projective_stabilizer_dimension"] == 2
    assert geometry["projective_orbit_dimension"] == 2
    assert geometry["vector_stabilizer"] == "U(1)_Q"
    assert geometry["vector_stabilizer_on_cover"] == "{(diag(z^3,z^-3),z): z in U(1)}"
    assert geometry["vector_stabilizer_dimension"] == 1
    assert geometry["goldstone_count"] == 3
    assert geometry["goldstone_count"] == geometry["broken_generator_count"]
    assert action["hypercharge_fixes_projective_ray"] is True
    assert action["t3_fixes_projective_ray"] is True
    assert action["hypercharge_fixes_vector"] is False
    assert action["t3_fixes_vector"] is False
    assert action["diagonal_q_fixes_vector"] is True
    assert "higgs_mass" in report["explicit_nonclaims"]
    assert report["physical_claim"] is False


def test_borel_weil_higgs_stabilizers_pass_arbitrary_nontrivial_phase_checks():
    for phase in (0.173, 0.731, 1.417):
        report = borel_weil_higgs_carrier_receipt({"stabilizer_test_phase_radians": phase})
        action = report["group_action_acceptance"]

        assert report["BOREL_WEIL_HIGGS_CARRIER_RECEIPT"] is True
        assert action["test_phase_radians"] == phase
        assert action["hypercharge_fixes_projective_ray"] is True
        assert action["t3_fixes_projective_ray"] is True
        assert action["hypercharge_fixes_vector"] is False
        assert action["t3_fixes_vector"] is False
        assert action["diagonal_q_fixes_vector"] is True


def test_borel_weil_higgs_carrier_blocks_quantitative_promotion():
    report = borel_weil_higgs_carrier_receipt({"derived_quantitative_claims": ["higgs_mass", "v"]})

    assert report["BOREL_WEIL_HIGGS_CARRIER_RECEIPT"] is False
    assert report["checks"]["forbidden_quantitative_promotions_absent"] is False
    assert report["promoted_forbidden_claims"] == ["higgs_mass", "v"]


def test_borel_weil_higgs_carrier_report_writer(tmp_path):
    report = write_borel_weil_higgs_carrier_report(tmp_path)
    payload_path = tmp_path / "borel_weil_higgs_carrier_report.json"
    markdown_path = tmp_path / "borel_weil_higgs_carrier_report.md"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert report["BOREL_WEIL_HIGGS_CARRIER_RECEIPT"] is True
    assert payload["carrier_identification"] == "H_OPH = H^0(CP1, O(1)) ~= C^2"
    assert payload["symmetry_breaking_geometry"]["projective_stabilizer_dimension"] == 2
    assert payload["symmetry_breaking_geometry"]["vector_stabilizer_dimension"] == 1
    assert markdown_path.exists()
    assert "Projective-ray stabilizer: (U(1)_T3 x U(1)_Y)/finite_center" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert "does not derive m_H" in markdown_path.read_text(encoding="utf-8")


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
