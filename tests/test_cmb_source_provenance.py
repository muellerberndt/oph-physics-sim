from __future__ import annotations

from oph_fpe.cosmology.source_provenance import (
    PROMOTED_CMB_SOURCE_QUANTITIES,
    certify_cmb_source_provenance,
)


def test_cmb_source_provenance_certificate_accepts_clean_source_graph():
    cert = certify_cmb_source_provenance(_clean_nodes(), _clean_reducers(), global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is True
    assert cert["blockers"] == []
    assert cert["N_CRC_consensus_invariant_receipt"] is True


def test_cmb_source_provenance_fails_closed_on_no_data_contradiction():
    nodes = _clean_nodes()
    nodes[0]["no_cmb_data_used"] = True
    nodes[0]["fit_to_planck"] = True

    cert = certify_cmb_source_provenance(nodes, _clean_reducers(), global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "contradictory_no_data_use_provenance:eta_R" in cert["blockers"]


def test_cmb_source_provenance_rejects_measurement_dependent_ancestor():
    nodes = _clean_nodes()
    nodes[0]["parents"] = ["planck_selector"]
    nodes.append(
        {
            "node_id": "planck_selector",
            "quantity": "model_selection",
            "source": "measurement_fit",
            "source_kind": "measurement_fit",
            "source_only": False,
            "no_cmb_data_used": False,
            "measurement_data_used": True,
            "parents": [],
        }
    )

    cert = certify_cmb_source_provenance(nodes, _clean_reducers(), global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert cert["TRANSITIVE_SOURCE_ANCESTRY_RECEIPT"] is False
    assert any("planck_selector" in blocker and "eta_R->planck_selector" in blocker for blocker in cert["blockers"])


def test_cmb_source_provenance_requires_pooled_or_single_global_reducers():
    reducers = _clean_reducers()
    reducers["B_A_k_a"] = {"mode": "shard_local_average"}

    cert = certify_cmb_source_provenance(_clean_nodes(), reducers, global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "B_A_k_a_source_reducer_not_pooled_or_global" in cert["blockers"]


def test_cmb_source_provenance_blocks_additive_N_CRC_without_disjoint_schema():
    reducers = _clean_reducers()
    reducers["N_CRC"] = {
        "mode": "additive_capacity",
        "consensus_invariant": False,
        "additive_capacity_schema": True,
        "disjoint_coverage_receipt": False,
    }

    cert = certify_cmb_source_provenance(_clean_nodes(), reducers, global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "N_CRC_not_consensus_invariant_or_additive_disjoint" in cert["blockers"]


def _clean_nodes() -> list[dict]:
    sources = {
        "eta_R": "finite_repair_transition_clock",
        "Gamma_rec": "finite_repair_transition_clock",
        "A_zeta": "finite_lattice",
        "q_IR": "scale_compressed_24_round_finite_ladder",
        "ell_IR": "scale_compressed_24_round_finite_ladder",
        "B_A_k_a": "parent_collar_finite_difference",
        "rho_A_a": "finite_lattice",
        "N_CRC": "OPH_screen_capacity_branch_predeclared",
    }
    return [
        {
            "node_id": quantity,
            "quantity": quantity,
            "source": sources[quantity],
            "source_kind": sources[quantity],
            "source_report": f"{quantity}_report",
            "source_only": True,
            "no_cmb_data_used": True,
            "fit_to_planck": False,
            "measurement_data_used": False,
            "parents": [],
        }
        for quantity in PROMOTED_CMB_SOURCE_QUANTITIES
    ]


def _clean_reducers() -> dict[str, dict]:
    reducers = {
        quantity: {
            "mode": "single_global_source",
            "single_global_source": True,
            "shard_local_nonlinear_average": False,
        }
        for quantity in PROMOTED_CMB_SOURCE_QUANTITIES
    }
    reducers["N_CRC"] = {
        "mode": "consensus_invariant",
        "consensus_invariant": True,
        "additive_capacity_schema": False,
        "disjoint_coverage_receipt": False,
    }
    return reducers


def _clean_global_checks() -> dict[str, object]:
    return {
        "official_likelihood_rollup": "global",
        "cdm_limit_rollup": "global",
        "HERMETIC_READ_SET_RECEIPT": True,
        "SOURCE_MODEL_FREEZE_RECEIPT": True,
    }
