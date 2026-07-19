from __future__ import annotations

from oph_fpe.cosmology.source_provenance import (
    PROMOTED_CMB_SOURCE_QUANTITIES,
    certify_cmb_source_provenance,
)


def test_cmb_source_provenance_holds_physical_n_closed_without_packet_replay():
    cert = certify_cmb_source_provenance(_clean_nodes(), _clean_reducers(), global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert cert["N_CRC_direct_public_record_capacity_receipt"] is False
    assert "N_CRC_direct_public_record_capacity_receipt_missing" in cert["blockers"]
    assert (
        cert["N_CRC_status"][
            "independent_public_record_capacity_recomputation_receipt"
        ]
        is False
    )


def test_cmb_source_provenance_rejects_truthy_string_capacity_declarations():
    reducers = _clean_reducers()
    reducers["N_CRC"] = {
        key: "false"
        for key in (
            "exact_public_record_capacity_evaluator",
            "complete_terminal_fiber_receipt",
            "whole_fiber_scalarization_receipt",
            "target_free_capacity_producer_receipt",
            "robust_closure_receipt",
            "unique_regulator_stable_slack_zero_receipt",
            "horizon_record_saturation_receipt",
            "physical_N_closure_receipt",
        )
    }

    cert = certify_cmb_source_provenance(
        _clean_nodes(), reducers, global_checks=_clean_global_checks()
    )

    assert cert["N_CRC_direct_public_record_capacity_receipt"] is False
    assert not any(
        value is True
        for key, value in cert["N_CRC_status"].items()
        if key.endswith("receipt")
    )


def test_cmb_source_provenance_rejects_truthy_string_subreceipts() -> None:
    nodes = _clean_nodes()
    for node in nodes:
        node["source_only"] = "false"
        node["no_cmb_data_used"] = "false"
    reducers = _clean_reducers()
    for quantity, reducer in reducers.items():
        if quantity != "N_CRC":
            reducer["single_global_source"] = "false"
    global_checks = _clean_global_checks()
    global_checks["HERMETIC_READ_SET_RECEIPT"] = "false"
    global_checks["SOURCE_MODEL_FREEZE_RECEIPT"] = "false"

    cert = certify_cmb_source_provenance(
        nodes,
        reducers,
        global_checks=global_checks,
    )

    assert cert["TRANSITIVE_SOURCE_ANCESTRY_RECEIPT"] is False
    assert cert["HERMETIC_READ_SET_RECEIPT"] is False
    assert cert["SOURCE_MODEL_FREEZE_RECEIPT"] is False
    assert cert["pooled_source_reducer_receipt"] is False


def test_cmb_source_provenance_bounds_long_parent_chains_without_recursion() -> None:
    nodes = []
    for index in range(513):
        nodes.append(
            {
                "node_id": f"node-{index}",
                "quantity": "eta_R" if index == 0 else "intermediate",
                "source": "finite_lattice",
                "source_kind": "finite_lattice",
                "source_only": True,
                "no_cmb_data_used": True,
                "parents": [f"node-{index + 1}"] if index < 512 else [],
            }
        )

    cert = certify_cmb_source_provenance(
        nodes,
        {},
        global_checks={},
        required_quantities=("eta_R",),
    )

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "provenance_node_budget_exceeded" in cert["blockers"]

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


def test_cmb_source_provenance_rejects_unregistered_clean_looking_ancestor():
    nodes = _clean_nodes()
    nodes[0]["parents"] = ["asserted_clean"]
    nodes.append(
        {
            "node_id": "asserted_clean",
            "quantity": "intermediate",
            "source": "caller_asserted_clean",
            "source_kind": "caller_asserted_clean",
            "source_only": True,
            "no_cmb_data_used": True,
            "parents": [],
        }
    )

    cert = certify_cmb_source_provenance(
        nodes,
        _clean_reducers(),
        global_checks=_clean_global_checks(),
    )

    assert cert["TRANSITIVE_SOURCE_ANCESTRY_RECEIPT"] is False
    assert any(
        blocker.startswith("unregistered_source_ancestor:asserted_clean")
        for blocker in cert["blockers"]
    )


def test_cmb_source_provenance_requires_pooled_or_single_global_reducers():
    reducers = _clean_reducers()
    reducers["B_A_k_a"] = {"mode": "shard_local_average"}

    cert = certify_cmb_source_provenance(_clean_nodes(), reducers, global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "B_A_k_a_source_reducer_not_pooled_or_global" in cert["blockers"]


def test_cmb_source_provenance_blocks_counts_without_direct_public_record_receipt():
    reducers = _clean_reducers()
    reducers["N_CRC"] = {
        "mode": "additive_capacity",
        "consensus_invariant": False,
        "additive_capacity_schema": True,
        "disjoint_coverage_receipt": False,
    }

    cert = certify_cmb_source_provenance(_clean_nodes(), reducers, global_checks=_clean_global_checks())

    assert cert["CMB_SOURCE_PROVENANCE_RECEIPT"] is False
    assert "N_CRC_direct_public_record_capacity_receipt_missing" in cert["blockers"]


def _clean_nodes() -> list[dict]:
    sources = {
        "eta_R": "finite_repair_transition_clock",
        "Gamma_rec": "finite_repair_transition_clock",
        "A_zeta": "finite_lattice",
        "q_IR": "scale_compressed_24_round_finite_ladder",
        "ell_IR": "scale_compressed_24_round_finite_ladder",
        "B_A_k_a": "parent_collar_finite_difference",
        "rho_A_a": "finite_lattice",
        "N_CRC": "OPH_direct_public_record_capacity",
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
        "mode": "direct_public_record_capacity",
        "exact_public_record_capacity_evaluator": True,
        "complete_terminal_fiber_receipt": True,
        "whole_fiber_scalarization_receipt": True,
        "target_free_capacity_producer_receipt": True,
        "robust_closure_receipt": True,
        "unique_regulator_stable_slack_zero_receipt": True,
        "horizon_record_saturation_receipt": True,
        "physical_N_closure_receipt": True,
    }
    return reducers


def _clean_global_checks() -> dict[str, object]:
    return {
        "official_likelihood_rollup": "global",
        "cdm_limit_rollup": "global",
        "HERMETIC_READ_SET_RECEIPT": True,
        "SOURCE_MODEL_FREEZE_RECEIPT": True,
    }
